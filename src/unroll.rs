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
    pub(crate) fn new(ctx: &Unroller, source: Option<&gltf::Node>) -> Result<Self> {
        let mut children = BTreeMap::new();

        // Add this variable here so we can have a root node reference later
        // on.
        let roots: Vec<gltf::Node>;

        let source = match source {
            Some(node) => node,
            None => {
                // At the root, if we have a single root node, that becomes
                // our starting node. If we have several, we create an empty
                // node at root and put them as children.

                roots = ctx.root_nodes();
                if roots.is_empty() {
                    bail!("Node: No root nodes found");
                } else if roots.len() == 1 {
                    // Single root node, the default case.
                    &roots[0]
                } else {
                    // Multiple root nodes, we need to wiggle things up for
                    // our format.

                    // XXX: For unknown reasons, doing the sensible thing here
                    // and generating an empty root node causes things to crap
                    // out when the scene is skeletized and loaded into
                    // Raylib. So instead we do things in a messier way and
                    // add the side nodes as additional children of the first
                    // node.
                    for sibling in roots.iter().skip(1) {
                        let name = ctx.node_names[sibling.index()].clone();
                        children.insert(name, Self::new(ctx, Some(sibling))?);
                    }
                    &roots[0]
                }
            }
        };

        // TODO: Support node cameras. Not high priority for model
        // work.
        for child in source.children() {
            let name = ctx.node_names[child.index()].clone();
            children.insert(name, Self::new(ctx, Some(&child))?);
        }

        let skin = if let Some(skin) = source.skin() {
            skin.joints()
                .map(|n| {
                    n.name()
                        .map(|n| n.to_string())
                        .ok_or_else(|| anyhow::anyhow!("load_skin: Joint has no name"))
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };

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

    /// Turn the scene graph of this node into a skeletal animation.
    pub fn skeletize(&mut self) {
        if self.1.contains_key("armature") {
            panic!("skeletize: Armature already exists");
        }

        // Clear root's animation and transformation
        self.0 .0.transform = None;
        self.0 .0.transform_matrix = None;
        self.0 .0.animations.clear();

        // Primitives from child nodes that are transformed to world space and
        // added to the skinned root node.
        let mut new_primitives = Vec::new();

        // Turn all child nodes with meshes into children of the armature. The
        // armature itself serves as the joint of the root mesh.

        let mut armature_data = self.0 .0.clone();

        // Root mesh might already have some skinned animation.
        if !armature_data.skin.is_empty() {
            // Then we won't try to bake a transformation into its
            assert!(
                armature_data.animations.is_empty(),
                "skeletize: Skinned root node must not have its own animations"
            );

            if !armature_data.get_transform().is_identity() {
                // TODO: Implement the reverse transformation on the
                // primitives of a skinned rootwe push to new_primitives.
                panic!("skeletize: Transform baking not supported for skinned root node");
                // This is doable, but an extra bit of complexity I'm not
                // bothering with yet...
            }

            // Retain original root mesh data as is instead of pushing it
            // through the remunging pipeline.
            new_primitives.append(&mut armature_data.mesh);
            armature_data.mesh = Vec::new();
            armature_data.skin.clear();
        }

        let mut armature = Node((armature_data,), BTreeMap::new());

        let animations = self.get_animation_names();

        for key in self.1.keys().cloned().collect::<Vec<_>>() {
            // XXX: This only looks at the first level of nodes for has mesh
            // (skeletonize it) / doesn't have mesh (maybe it's already a
            // joint, keep it as is). Weird models might have more complex
            // nesting of mesh vs non-mesh modes but I can't bother to figure
            // out them all here.
            if self.1[&key].mesh.is_empty() {
                log::info!("skeletize: Skipping child node {key} with no mesh");
            }

            // Move mesh-carrying child nodes from root into armature.
            armature.1.insert(key.clone(), self.1.remove(&key).unwrap());
        }

        // Build transformation matrices from global space to nodes.
        let mut transforms = Vec::new();

        for (_, node, parent_idx) in NodeIter::new("armature", &armature) {
            let mut transform = node.get_transform();

            if let Some(parent_idx) = parent_idx {
                transform = transforms[parent_idx] * transform;
            }
            transforms.push(transform);
        }

        for (i, (name, node, _)) in NodeIterMut::new("armature", &mut armature).enumerate() {
            if node.mesh.is_empty() {
                continue;
            }

            assert!(
                node.skin.is_empty(),
                "skeletize: Trying to process skinned child node"
            );

            self.0 .0.skin.push(name.clone());
            if self.0 .0.skin.len() > 254 {
                panic!("skeletize: Too many joints in armature");
            }

            let joint_idx = (self.0 .0.skin.len() - 1) as u8;

            for mut p in node.mesh.drain(..) {
                p.transform(&transforms[i]);
                p.splat_joint(joint_idx);
                new_primitives.push(p);
            }

            // We need to add neutral animations to unanimated child nodes so
            // that Raylib will show them transformed.
            if !node.get_transform().is_identity() {
                node.add_neutral_animations(&animations);
            }
        }

        self.1.insert("armature".to_string(), armature);
        self.0 .0.mesh = new_primitives;
    }

    /// Get names and lenghts of all animations in this node tree.
    fn get_animation_names(&self) -> BTreeMap<String, f32> {
        let mut ret = BTreeMap::new();
        for (_, node, _) in NodeIter::new("", self) {
            for (name, anim) in &node.animations {
                ret.insert(name.clone(), anim.duration());
            }
        }
        ret
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
    type Item = (String, &'a NodeData, Option<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        let (name, node, parent) = self.stack.pop()?;
        self.stack.extend(
            node.1
                .iter()
                .map(|(name, c)| (name.clone(), c, Some(self.current_idx))),
        );
        self.current_idx += 1;
        Some((name, &node.0 .0, parent))
    }
}

pub struct NodeIterMut<'a> {
    stack: Vec<(String, *mut Node, Option<usize>)>,
    current_idx: usize,
    phantom: std::marker::PhantomData<&'a mut Node>,
}

impl<'a> NodeIterMut<'a> {
    pub fn new(root_name: impl Into<String>, root: &'a mut Node) -> Self {
        Self {
            stack: vec![(root_name.into(), root, None)],
            current_idx: 0,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<'a> Iterator for NodeIterMut<'a> {
    type Item = (String, &'a mut NodeData, Option<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        let (name, node, parent) = self.stack.pop()?;

        let node = unsafe {
            for (name, c) in (*node).1.iter_mut() {
                self.stack.push((name.clone(), c, Some(self.current_idx)));
            }
            &mut (*node).0 .0
        };
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
    pub skin: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub mesh: Vec<Primitive>,
    // Animation implicitly attached to one node.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub animations: BTreeMap<String, Animation>,
}

impl NodeData {
    pub fn get_transform(&self) -> Mat4 {
        assert!(
            self.transform.is_none() || self.transform_matrix.is_none(),
            "NodeData: Two transforms specified"
        );
        if let Some(transform) = self.transform.as_ref() {
            Mat4::from(*transform)
        } else {
            self.transform_matrix.unwrap_or_default()
        }
    }

    /// Add dummy animations that apply translation, needed by skeletize.
    pub fn add_neutral_animations(&mut self, animations: &BTreeMap<String, f32>) {
        let translation = self
            .transform
            .as_ref()
            .map(|t| t.translation)
            .unwrap_or_default();
        for (name, duration) in animations {
            if self.animations.contains_key(name) {
                continue;
            }
            self.animations
                .insert(name.clone(), Animation::neutral(*duration, translation));
        }
    }
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
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

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
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

impl From<Trs> for Mat4 {
    fn from(trs: Trs) -> Self {
        Mat4(glam::Mat4::from_scale_rotation_translation(
            trs.scale,
            trs.rotation.into(),
            trs.translation,
        ))
    }
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

    /// Apply transformation to the spatial geometry of the primitive.
    fn transform(&mut self, transform: &Mat4) {
        for p in self.positions.iter_mut() {
            *p = transform.transform_point3(*p);
        }
        for n in self.normals.iter_mut() {
            *n = transform.transform_vector3(*n);
            *n = n.normalize();
        }
    }

    /// Bind the entire primitive fully to the single joint.
    fn splat_joint(&mut self, joint: u8) {
        self.joints = (0..self.positions.len())
            .map(|_| [joint, 0, 0, 0])
            .collect();

        self.weights = (0..self.positions.len())
            .map(|_| Vec4::new(1.0, 0.0, 0.0, 0.0))
            .collect();
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

    pub fn neutral(duration: f32, translation: Vec3) -> Self {
        Self {
            translation: vec![(0.0, translation), (duration, translation)],
            rotation: vec![(0.0, Angle::default()), (duration, Angle::default())],
            scale: vec![(0.0, Vec3::ONE), (duration, Vec3::ONE)],
            weight: vec![(0.0, 1.0), (duration, 1.0)],
        }
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

    pub fn duration(&self) -> f32 {
        let mut max_time = 0.0f32;
        if let Some((t, _)) = self.translation.last() {
            max_time = max_time.max(*t);
        }
        if let Some((t, _)) = self.rotation.last() {
            max_time = max_time.max(*t);
        }
        if let Some((t, _)) = self.scale.last() {
            max_time = max_time.max(*t);
        }
        if let Some((t, _)) = self.weight.last() {
            max_time = max_time.max(*t);
        }
        max_time
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

    pub fn root_nodes(&self) -> Vec<gltf::Node> {
        let children: HashSet<usize> = self
            .gltf
            .nodes()
            .flat_map(|n| n.children().map(|n| n.index()))
            .collect();
        self.gltf
            .nodes()
            .filter(|n| !children.contains(&n.index()))
            .collect()
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

    pub fn inverse(&self) -> Self {
        Self(self.0.inverse())
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

impl std::ops::Mul for Mat4 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0)
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
