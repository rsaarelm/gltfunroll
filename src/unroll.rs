use std::{
    collections::{BTreeMap, HashSet},
    path::Path,
};

use anyhow::{bail, Result};
use glam::{Quat, Vec2, Vec3, Vec4};
use gltf::animation::util::ReadOutputs;
use serde::{Deserialize, Serialize};

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

        let skin = source
            .skin()
            .map(|s| Joint::load_skin(ctx, &s))
            .unwrap_or_else(|| Ok(Default::default()))?;

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
                    rotation: Angle::from_array(rotation),
                    scale: Vec3::from(scale),
                });
            }
        }

        // XXX: We're ignoring morph targets and morph target weights for now.

        let mesh = if let Some(m) = source.mesh() {
            m.primitives()
                .map(|p| Primitive::new(ctx, &p))
                .collect::<Result<Vec<_>>>()?
        } else {
            Default::default()
        };

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
#[serde(default, rename_all = "kebab-case")]
pub struct NodeData {
    #[serde(skip_serializing_if = "Trs::is_empty")]
    pub transform: Option<Trs>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transform_matrix: Option<Mat4>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub camera: Option<Camera>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub skin: Vec<Joint>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub mesh: Vec<Primitive>,
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
    #[serde(skip_serializing_if = "Angle::is_neutral")]
    pub rotation: Angle,
    #[serde(skip_serializing_if = "Trs::empty_scale")]
    pub scale: Vec3,
}

impl Trs {
    fn is_empty(trs: &Option<Self>) -> bool {
        trs.as_ref().map_or(true, |t| {
            t.translation == Vec3::ZERO && t.rotation.is_neutral() && t.scale == Vec3::ONE
        })
    }

    fn empty_translation(vec: &Vec3) -> bool {
        *vec == Vec3::ZERO
    }

    fn empty_scale(vec: &Vec3) -> bool {
        *vec == Vec3::ONE
    }
}

impl Default for Trs {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation: Angle::default(),
            scale: Vec3::ONE,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(
    from = "((String,), InverseBindMatrix)",
    into = "((String,), InverseBindMatrix)"
)]
pub struct Joint {
    pub name: String,
    pub inverse_bind_matrix: Mat4,
}

impl Joint {
    pub(crate) fn new(name: String, inverse_bind_matrix: Mat4) -> Self {
        Self {
            name,
            inverse_bind_matrix,
        }
    }

    pub(crate) fn load_skin(ctx: &Unroller, source: &gltf::Skin) -> Result<Vec<Self>> {
        let joints: Vec<String> = source
            .joints()
            .map(|n| {
                n.name()
                    .map(|n| n.to_string())
                    .ok_or_else(|| anyhow::anyhow!("Skin: Joint has no name"))
            })
            .collect::<Result<Vec<String>>>()?;

        let inverse_bind_matrices: Vec<Mat4> = if let Some(ibm) = source
            .reader(|buffer| ctx.buffers.get(buffer.index()).map(|b| b.0.as_slice()))
            .read_inverse_bind_matrices()
        {
            ibm.map(Mat4::from).collect()
        } else {
            Default::default()
        };

        Ok(joints
            .into_iter()
            .enumerate()
            .map(|(i, name)| {
                Self::new(
                    name,
                    inverse_bind_matrices.get(i).copied().unwrap_or_default(),
                )
            })
            .collect())
    }
}

impl From<((String,), InverseBindMatrix)> for Joint {
    fn from(((name,), inverse_bind_matrix): ((String,), InverseBindMatrix)) -> Self {
        Self::new(name, inverse_bind_matrix.inverse_bind_matrix)
    }
}

impl From<Joint> for ((String,), InverseBindMatrix) {
    fn from(joint: Joint) -> Self {
        (
            (joint.name,),
            InverseBindMatrix {
                inverse_bind_matrix: joint.inverse_bind_matrix,
            },
        )
    }
}

// Dummy struct to make IDM layout nicer.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "kebab-case")]
struct InverseBindMatrix {
    #[serde(skip_serializing_if = "Mat4::is_identity")]
    inverse_bind_matrix: Mat4,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "kebab-case")]
pub struct Primitive {
    // Make these into chunks just so the IDM document does not end up with a
    // giant line.
    pub indices: Vec<Vec<u16>>,

    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub tex_coords: Vec<Vec2>,
    pub joints: Vec<[u8; 4]>,
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
            .map(|i| i as u16)
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
            joints = j
                .into_u16()
                .map(|[x, y, z, w]| [x as u8, y as u8, z as u8, w as u8])
                .collect();
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
    pub rotation: Vec<(f32, Angle)>,
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
                    .map(|(t, s)| (s, Angle::from_array(t)))
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

/// Custom wrapper type for Mat4, mostly so that we get a nice IDM
/// serialization as four rows via the nested arrays serialization type.
#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(from = "[[f32; 4]; 4]", into = "[[f32; 4]; 4]")]
pub struct Mat4(pub glam::Mat4);

impl Mat4 {
    pub fn is_identity(&self) -> bool {
        self.0 == glam::Mat4::IDENTITY
    }
}

impl From<[[f32; 4]; 4]> for Mat4 {
    fn from(m: [[f32; 4]; 4]) -> Self {
        Self(glam::Mat4::from_cols_array_2d(&m))
    }
}

impl From<Mat4> for [[f32; 4]; 4] {
    fn from(m: Mat4) -> Self {
        m.0.to_cols_array_2d()
    }
}

impl std::ops::Deref for Mat4 {
    type Target = glam::Mat4;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Angle {
    /// Quaternion. The recommended standard for storing rotations, but not at
    /// all human-understandable.
    Quat(Quat),

    /// Euler angles for yaw, pitch and roll rotations, given in degrees. A
    /// variant format for humans writing angles.
    Euler(f32, f32, f32),
}

impl Angle {
    pub fn from_array(a: [f32; 4]) -> Self {
        Self::Quat(Quat::from_array(a))
    }

    pub fn to_array(&self) -> [f32; 4] {
        match self {
            Angle::Quat(q) => q.to_array(),
            Angle::Euler(_, _, _) => Quat::from(*self).to_array(),
        }
    }

    pub fn is_neutral(&self) -> bool {
        match self {
            Angle::Euler(x, y, z) => *x == 0.0 && *y == 0.0 && *z == 0.0,
            Angle::Quat(q) => *q == Quat::IDENTITY,
        }
    }
}

impl Default for Angle {
    fn default() -> Self {
        Self::Quat(Quat::IDENTITY)
    }
}

impl From<Quat> for Angle {
    fn from(q: Quat) -> Self {
        Self::Quat(q)
    }
}

impl From<Angle> for Quat {
    fn from(a: Angle) -> Self {
        match a {
            Angle::Quat(q) => q,
            // Convert the degrees in v to radians.
            Angle::Euler(x, y, z) => glam::Quat::from_euler(
                glam::EulerRot::YXZ,
                y.to_radians(),
                x.to_radians(),
                z.to_radians(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn angle_euler() {
        assert_eq!(
            Quat::from(Angle::Euler(90.0, 0.0, 0.0)),
            Quat::from_rotation_x(90.0f32.to_radians())
        );
        assert_eq!(
            Quat::from(Angle::Euler(0.0, 90.0, 0.0)),
            Quat::from_rotation_y(90.0f32.to_radians())
        );
        assert_eq!(
            Quat::from(Angle::Euler(0.0, 0.0, 90.0)),
            Quat::from_rotation_z(90.0f32.to_radians())
        );
    }
}
