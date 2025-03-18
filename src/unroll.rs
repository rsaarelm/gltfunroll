use std::{
    collections::{BTreeMap, HashSet},
    path::Path,
};

use anyhow::{bail, Result};
use glam::{Quat, Vec2, Vec3, Vec4};
use gltf::animation::util::ReadOutputs;
use serde::{Deserialize, Serialize};

use crate::{gensym, Mat4};

// Because the output is IDM, the contents of the node are wrapped in this
// struct that expresses the tree structure. Actual contents, other than names
// and children, are stored in NodeData. Serializing Nodes will produce nice
// IDM files.

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Node(pub (NodeData,), pub BTreeMap<String, Node>);

impl Node {
    pub(crate) fn new(ctx: &Unroller, source: &gltf::Node) -> Result<Self> {
        // TODO: Support node cameras. Not high priority for model
        // work.
        let mut children = BTreeMap::new();
        for child in source.children() {
            let name = child.name().map_or_else(crate::gensym, |n| n.to_string());
            children.insert(name, Self::new(ctx, &child)?);
        }

        let skin = source.skin().map(|s| Skin::new(ctx, &s)).transpose()?;

        // Flatten matrix/TRS transform into Node attribute level.
        let mut transform = None;
        let mut transform_matrix = None;
        match source.transform() {
            gltf::scene::Transform::Matrix { matrix } => {
                transform_matrix = Some(crate::Mat4::from(matrix));
            }
            gltf::scene::Transform::Decomposed {
                translation,
                rotation,
                scale,
            } => {
                transform = Some(Trs {
                    translation: Vec3::from(translation),
                    rotation: Quat::from_array(rotation),
                    scale: Vec3::from(scale),
                });
            }
        }

        // XXX: We're ignoring morph targets and morph target weights for now.

        let mesh = source
            .mesh()
            .map(|m| {
                m.primitives()
                    .map(|p| Primitive::new(ctx, &p))
                    .collect::<Result<_>>()
            })
            .transpose()?
            .unwrap_or_default();

        let mut animations = BTreeMap::new();
        for a in ctx.gltf.animations() {
            // Animations that affect multiple nodes are split into per-node
            // channel sets, does this make sense?
            //
            // We can hopefully merge the animations back during roll stage
            // based on names.
            let channels = a
                .channels()
                .filter(|c| c.target().node().index() == source.index())
                .collect::<Vec<_>>();
            if channels.is_empty() {
                continue;
            }

            let name = a.name().map_or_else(gensym, |n| n.to_string());
            let channels: Vec<Channel> = a
                .channels()
                .map(|c| Channel::new(ctx, &c))
                .collect::<Result<_>>()?;
            animations.insert(name, channels);
        }

        Ok(Node(
            (NodeData {
                mesh,
                skin,
                transform,
                transform_matrix,
                animations,
                ..Default::default()
            },),
            children,
        ))
    }
}

impl std::ops::Deref for Node {
    type Target = NodeData;

    fn deref(&self) -> &Self::Target {
        &self.0 .0
    }
}

pub struct NodeIter<'a> {
    stack: Vec<(String, &'a Node, Option<usize>)>,
    current_idx: usize,
}

impl<'a> NodeIter<'a> {
    pub fn new(root_name: impl Into<String>, root: &'a Node) -> Self {
        Self {
            stack: vec![(root_name.into(), root, None)],
            current_idx: 0,
        }
    }
}

impl<'a> Iterator for NodeIter<'a> {
    type Item = (String, &'a Node, Option<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        let (name, node, parent) = self.stack.pop()?;
        self.stack.extend(
            node.1
                .iter()
                .map(|(name, c)| (name.clone(), c, Some(self.current_idx))),
        );
        self.current_idx += 1;
        Some((name, node, parent))
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct NodeData {
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub mesh: Vec<Primitive>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skin: Option<Skin>,
    #[serde(skip_serializing_if = "Trs::is_empty")]
    pub transform: Option<Trs>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transform_matrix: Option<Mat4>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub camera: Option<Camera>,
    // Animation implicitly attached to one node.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub animations: BTreeMap<String, Vec<Channel>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Camera {
    Perspective {
        aspect_ratio: f32,
        yfov: f32,
        znear: f32,
        zfar: f32,
    },
    Orthographic {
        xmag: f32,
        ymag: f32,
        znear: f32,
        zfar: f32,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Trs {
    #[serde(skip_serializing_if = "Trs::empty_translation")]
    pub translation: Vec3,
    #[serde(skip_serializing_if = "Trs::empty_rotation")]
    pub rotation: Quat,
    #[serde(skip_serializing_if = "Trs::empty_scale")]
    pub scale: Vec3,
}

impl Trs {
    fn is_empty(trs: &Option<Self>) -> bool {
        trs.as_ref().map_or(true, |t| {
            t.translation == Vec3::ZERO && t.rotation == Quat::IDENTITY && t.scale == Vec3::ONE
        })
    }

    fn empty_translation(vec: &Vec3) -> bool {
        *vec == Vec3::ZERO
    }

    fn empty_rotation(quat: &Quat) -> bool {
        *quat == Quat::IDENTITY
    }

    fn empty_scale(vec: &Vec3) -> bool {
        *vec == Vec3::ONE
    }
}

impl Default for Trs {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct Skin {
    pub joints: Vec<String>,
    pub inverse_bind_matrices: Vec<Mat4>,
}

impl Skin {
    pub(crate) fn new(ctx: &Unroller, source: &gltf::Skin) -> Result<Self> {
        let joints: Vec<String> = source
            .joints()
            .map(|n| {
                n.name()
                    .map(|n| n.to_string())
                    .ok_or_else(|| anyhow::anyhow!("Skin: Joint has no name"))
            })
            .collect::<Result<Vec<String>>>()?;

        let inverse_bind_matrices = source
            .reader(|buffer| ctx.buffers.get(buffer.index()).map(|b| b.0.as_slice()))
            .read_inverse_bind_matrices()
            .expect("Skin: No inverse bind matrices")
            .map(Mat4::from)
            .collect();

        Ok(Self {
            joints,
            inverse_bind_matrices,
        })
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "kebab-case")]
pub struct Primitive {
    // Make these into chunks just so the IDM document does not end up with a
    // giant line.
    pub indices: Vec<Vec<u32>>,

    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub tex_coords: Vec<Vec2>,
    pub joints: Vec<[u16; 4]>,
    pub weights: Vec<Vec4>,

    pub material: Material,
}

impl Primitive {
    pub(crate) fn new(ctx: &Unroller, primitive: &gltf::Primitive) -> Result<Self> {
        let reader =
            primitive.reader(|buffer| ctx.buffers.get(buffer.index()).map(|b| b.0.as_slice()));

        let indices = reader
            .read_indices()
            .expect("Primitive: No indices")
            .into_u32()
            .collect::<Vec<_>>();
        let indices = indices.chunks(3).map(|c| c.to_vec()).collect();

        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut tex_coords = Vec::new();
        let mut joints = Vec::new();
        let mut weights = Vec::new();

        if let Some(p) = reader.read_positions() {
            positions = p.map(Vec3::from).collect();
        }

        if let Some(n) = reader.read_normals() {
            normals = n.map(Vec3::from).collect();
        }

        if let Some(t) = reader.read_tex_coords(0) {
            tex_coords = t.into_f32().map(Vec2::from).collect();
        }

        if let Some(j) = reader.read_joints(0) {
            joints = j.into_u16().collect();
        }

        if let Some(w) = reader.read_weights(0) {
            weights = w.into_f32().map(Vec4::from).collect();
        }

        Ok(Primitive {
            indices,
            positions,
            normals,
            tex_coords,
            joints,
            weights,
            ..Default::default()
        })
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "kebab-case")]
pub struct Material {
    pub texture_map: Option<String>,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Channel {
    pub interpolation: Interpolation,
    pub path: AnimPath,
}

impl Channel {
    pub(crate) fn new(ctx: &Unroller, channel: &gltf::animation::Channel) -> Result<Self> {
        let interpolation = channel.sampler().interpolation().into();
        let reader =
            channel.reader(|buffer| ctx.buffers.get(buffer.index()).map(|b| b.0.as_slice()));
        let timestamps: Vec<f32> = reader
            .read_inputs()
            .expect("Channel: No timestamps")
            .collect();
        let path = match reader.read_outputs().expect("Channel: No outputs") {
            ReadOutputs::Translations(translations) => {
                let translations: Vec<(f32, Vec3)> = translations
                    .zip(timestamps.iter().copied())
                    .map(|(t, s)| (s, Vec3::from(t)))
                    .collect();
                AnimPath::Translation(translations)
            }
            ReadOutputs::Rotations(rotations) => {
                let rotations: Vec<(f32, Quat)> = rotations
                    .into_f32()
                    .zip(timestamps.iter().copied())
                    .map(|(t, s)| (s, Quat::from_array(t)))
                    .collect();
                AnimPath::Rotation(rotations)
            }
            ReadOutputs::Scales(scales) => {
                let scales: Vec<(f32, Vec3)> = scales
                    .zip(timestamps.iter().copied())
                    .map(|(t, s)| (s, Vec3::from(t)))
                    .collect();
                AnimPath::Scale(scales)
            }
            ReadOutputs::MorphTargetWeights(weights) => {
                let weights: Vec<(f32, f32)> = weights
                    .into_f32()
                    .zip(timestamps.iter().copied())
                    .map(|(t, s)| (s, t))
                    .collect();
                AnimPath::Weights(weights)
            }
        };

        Ok(Self {
            interpolation,
            path,
        })
    }
}

#[derive(Copy, Clone, Debug, Default, Serialize, Deserialize)]
pub enum Interpolation {
    #[default]
    Linear,
    Step,
    CubicSpline,
}

impl From<gltf::animation::Interpolation> for Interpolation {
    fn from(i: gltf::animation::Interpolation) -> Self {
        match i {
            gltf::animation::Interpolation::Linear => Self::Linear,
            gltf::animation::Interpolation::Step => Self::Step,
            gltf::animation::Interpolation::CubicSpline => Self::CubicSpline,
        }
    }
}

impl From<Interpolation> for gltf::animation::Interpolation {
    fn from(i: Interpolation) -> Self {
        match i {
            Interpolation::Linear => gltf::animation::Interpolation::Linear,
            Interpolation::Step => gltf::animation::Interpolation::Step,
            Interpolation::CubicSpline => gltf::animation::Interpolation::CubicSpline,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AnimPath {
    Translation(Vec<(f32, Vec3)>),
    Rotation(Vec<(f32, Quat)>),
    Scale(Vec<(f32, Vec3)>),
    Weights(Vec<(f32, f32)>),
}

impl Default for AnimPath {
    fn default() -> Self {
        Self::Translation(Vec::new())
    }
}

pub(crate) struct Unroller {
    pub gltf: gltf::Document,
    pub buffers: Vec<gltf::buffer::Data>,
}

impl Unroller {
    pub fn new(path: impl AsRef<Path>) -> Result<Unroller> {
        let (gltf, buffers, _images) = gltf::import(path)?;
        Ok(Unroller { gltf, buffers })
    }

    pub fn root_node(&self) -> Result<gltf::Node> {
        let children: HashSet<usize> = self
            .gltf
            .nodes()
            .flat_map(|n| n.children().map(|n| n.index()))
            .collect();
        let mut root_nodes = self.gltf.nodes().filter(|n| !children.contains(&n.index()));
        let Some(root_node) = root_nodes.next() else {
            bail!("No root node found")
        };
        if root_nodes.next().is_some() {
            bail!("Multiple root nodes found");
        }
        Ok(root_node)
    }
}
