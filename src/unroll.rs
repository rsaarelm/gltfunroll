use std::{
    collections::{BTreeMap, HashSet},
    path::Path,
};

use anyhow::{bail, Result};
use glam::{Quat, Vec2, Vec3, Vec4};
use gltf::animation::util::ReadOutputs;
use serde::{Deserialize, Serialize};

use crate::Mat4;

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
            let name = ctx.node_names[child.index()].clone();
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
            let name = ctx.anim_names[a.index()].clone();

            // Split out bits of animation that apply to this node and collect
            // them into the Animation object.
            let mut animation = Animation::default();
            for c in a.channels() {
                if c.target().node().index() != source.index() {
                    continue;
                }
                animation.add_channel(ctx, &c)?;
            }

            if !animation.is_empty() {
                animations.insert(name, animation);
            }
        }

        Ok(Node(
            (NodeData {
                mesh,
                skin,
                transform,
                transform_matrix,
                animations,
                camera: Default::default(),
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
    pub animations: BTreeMap<String, Animation>,
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

    #[serde(skip_serializing_if = "Material::is_empty")]
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

        let material = Material::new(&primitive.material())?;

        Ok(Primitive {
            indices,
            positions,
            normals,
            tex_coords,
            joints,
            weights,
            material,
        })
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "kebab-case")]
pub struct Material {
    pub base_color_texture: Option<String>,
    pub metallic_roughness_texture: Option<String>,
}

// Add rest of the material stuff, texture samplers etc. if needed. For now
// assume default values are good.

impl Material {
    pub(crate) fn new(material: &gltf::Material) -> Result<Self> {
        let pbr = material.pbr_metallic_roughness();

        let base_color_texture =
            pbr.base_color_texture()
                .and_then(|t| match t.texture().source().source() {
                    gltf::image::Source::View { .. } => None,
                    gltf::image::Source::Uri { uri, .. } => Some(uri.to_string()),
                });

        let metallic_roughness_texture =
            pbr.metallic_roughness_texture()
                .and_then(|t| match t.texture().source().source() {
                    gltf::image::Source::View { .. } => None,
                    gltf::image::Source::Uri { uri, .. } => Some(uri.to_string()),
                });

        Ok(Self {
            base_color_texture,
            metallic_roughness_texture,
        })
    }

    pub fn is_empty(&self) -> bool {
        self.base_color_texture.is_none() && self.metallic_roughness_texture.is_none()
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Animation {
    pub translation: Vec<(f32, Vec3)>,
    pub rotation: Vec<(f32, Quat)>,
    pub scale: Vec<(f32, Vec3)>,
    pub weight: Vec<(f32, f32)>,
}

impl Animation {
    pub fn is_empty(&self) -> bool {
        self.translation.is_empty()
            && self.rotation.is_empty()
            && self.scale.is_empty()
            && self.weight.is_empty()
    }

    pub(crate) fn add_channel(
        &mut self,
        ctx: &Unroller,
        channel: &gltf::animation::Channel,
    ) -> Result<()> {
        let reader =
            channel.reader(|buffer| ctx.buffers.get(buffer.index()).map(|b| b.0.as_slice()));
        let timestamps: Vec<f32> = reader
            .read_inputs()
            .expect("Channel: No timestamps")
            .collect();

        match reader.read_outputs().expect("Channel: No outputs") {
            ReadOutputs::Translations(translations) => {
                self.translation = translations
                    .zip(timestamps.iter().copied())
                    .map(|(t, s)| (s, Vec3::from(t)))
                    .collect()
            }

            ReadOutputs::Rotations(rotations) => {
                self.rotation = rotations
                    .into_f32()
                    .zip(timestamps.iter().copied())
                    .map(|(t, s)| (s, Quat::from_array(t)))
                    .collect()
            }

            ReadOutputs::Scales(scales) => {
                self.scale = scales
                    .zip(timestamps.iter().copied())
                    .map(|(t, s)| (s, Vec3::from(t)))
                    .collect()
            }

            ReadOutputs::MorphTargetWeights(weights) => {
                self.weight = weights
                    .into_f32()
                    .zip(timestamps.iter().copied())
                    .map(|(t, s)| (s, t))
                    .collect()
            }
        }

        Ok(())
    }
}

pub(crate) struct Unroller {
    gltf: gltf::Document,
    buffers: Vec<gltf::buffer::Data>,

    /// Original or generated names for all animations.
    anim_names: Vec<String>,
    /// Original or generated names for all meshes.
    node_names: Vec<String>,
}

impl Unroller {
    pub fn new(path: impl AsRef<Path>) -> Result<Unroller> {
        let (gltf, buffers, _images) = gltf::import(path)?;

        // Anyone who puts actual names in their data that end with '-gensym'
        // deserves what they get.
        let anim_names = gltf
            .animations()
            .enumerate()
            .map(|(i, a)| {
                a.name()
                    .map_or_else(|| format!("anim-{}-gensym", i), |n| n.to_string())
            })
            .collect();
        let node_names = gltf
            .nodes()
            .enumerate()
            .map(|(i, a)| {
                a.name()
                    .map_or_else(|| format!("node-{}-gensym", i), |n| n.to_string())
            })
            .collect();

        Ok(Unroller {
            gltf,
            buffers,
            anim_names,
            node_names,
        })
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
