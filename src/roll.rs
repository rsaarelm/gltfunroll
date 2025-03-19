use std::{
    any::TypeId,
    collections::{BTreeMap, HashMap},
};

use glam::{Quat, Vec2, Vec3, Vec4};
use gltf_json::{
    self as json,
    accessor::{ComponentType, GenericComponentType, Type},
    scene::UnitQuaternion,
    validation::{Checked, USize64},
};

use crate::{
    unroll::NodeIter, Animation, Mat4, Material, Node, NodeData, Primitive, Skin, VERSION,
};

/// Context object for rebuilding a glTF file from a `Node` tree.
#[derive(Clone, Debug)]
pub(crate) struct Roller {
    /// Name of the file and the first node.
    pub root_name: String,

    /// Map of data buffers that were already inserted, use for deduplication.
    pub buffers: HashMap<(TypeId, Vec<u8>), usize>,
    pub names: BTreeMap<String, usize>,
    pub textures: BTreeMap<String, usize>,

    pub accessors: Vec<json::Accessor>,
    pub animations: Vec<json::Animation>,
    pub buffer: Vec<u8>,
    pub buffer_views: Vec<json::buffer::View>,
    pub materials: Vec<json::Material>,
    pub meshes: Vec<json::Mesh>,
    pub nodes: Vec<json::Node>,
    pub skins: Vec<json::Skin>,
}

impl From<Roller> for json::Root {
    fn from(ctx: Roller) -> Self {
        let num_textures = ctx.textures.len();

        json::Root {
            accessors: ctx.accessors,
            animations: ctx.animations,
            asset: json::Asset {
                generator: Some(format!("gltfunroll v{}", VERSION.unwrap_or("unknown"))),
                version: "2.0".to_string(),
                ..Default::default()
            },
            buffer_views: ctx.buffer_views.clone(),
            buffers: vec![json::Buffer {
                byte_length: USize64::from(ctx.buffer.len()),
                uri: Some(format!("{}.bin", ctx.root_name)),
                name: None,
                extensions: Default::default(),
                extras: Default::default(),
            }],
            cameras: Default::default(),

            extensions: Default::default(),
            extensions_used: Default::default(),
            extensions_required: Default::default(),
            extras: Default::default(),

            // Push the collected textures in index order into the image list.
            images: {
                let mut images = ctx.textures.into_iter().collect::<Vec<_>>();
                images.sort_by_key(|(_, idx)| *idx);
                images
                    .into_iter()
                    .map(|(name, _)| json::Image {
                        buffer_view: None,
                        mime_type: None,
                        uri: Some(name),
                        name: None,
                        extensions: Default::default(),
                        extras: Default::default(),
                    })
                    .collect()
            },

            materials: ctx.materials,
            meshes: ctx.meshes,
            nodes: ctx.nodes,

            samplers: Default::default(),

            // Default scene that has the zeroth node as the root.
            scene: Some(json::Index::new(0)),
            scenes: vec![json::Scene {
                extensions: Default::default(),
                extras: Default::default(),
                name: None,
                nodes: vec![json::Index::new(0)],
            }],

            skins: ctx.skins,

            // One texture for each image, no extra data needed here.
            textures: (0..num_textures)
                .map(|i| json::Texture {
                    source: json::Index::new(i as u32),

                    sampler: Default::default(),
                    extensions: Default::default(),
                    extras: Default::default(),
                    name: Default::default(),
                })
                .collect(),
        }
    }
}

impl Roller {
    /// Build the roller, initialize name lookup for the node tree.
    pub fn new(root_name: impl Into<String>, root: &Node) -> Self {
        let root_name = root_name.into();
        let mut names = BTreeMap::new();
        for (i, (name, _, _)) in NodeIter::new(root_name.clone(), root).enumerate() {
            names.insert(name, i);
        }

        Self {
            root_name,

            buffers: Default::default(),
            names,
            textures: Default::default(),

            accessors: Default::default(),
            animations: Default::default(),
            buffer: Default::default(),
            buffer_views: Default::default(),
            materials: Default::default(),
            meshes: Default::default(),
            nodes: Default::default(),
            skins: Default::default(),
        }
    }

    pub fn push_data<E: BufferValue>(&mut self, data: &[E]) -> usize {
        self.push_data_inner(data, false, None)
    }

    pub fn push_input_data(&mut self, data: &[f32]) -> usize {
        self.push_data_inner(data, true, None)
    }

    pub fn push_index_data(&mut self, data: &[u16]) -> usize {
        self.push_data_inner(data, false, Some(json::buffer::Target::ElementArrayBuffer))
    }

    pub fn push_position_data(&mut self, data: &[Vec3]) -> usize {
        self.push_data_inner(data, true, Some(json::buffer::Target::ArrayBuffer))
    }

    pub fn push_attribute_data<E: BufferValue>(&mut self, data: &[E]) -> usize {
        self.push_data_inner(data, false, Some(json::buffer::Target::ArrayBuffer))
    }

    /// Push a buffer of data into the output glTF.
    ///
    /// Return the accessor index for the data.
    fn push_data_inner<E: BufferValue>(
        &mut self,
        data: &[E],
        compute_bounds: bool,
        target: Option<json::buffer::Target>,
    ) -> usize {
        assert!(!data.is_empty(), "Pushing empty data array");

        // Get raw byte data.
        let raw_data: Vec<u8> = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
                .to_vec()
        };
        let type_id = TypeId::of::<E>();

        // Check if the buffer is already present.
        if let Some(idx) = self.buffers.get(&(type_id, raw_data.clone())) {
            return *idx;
        } else {
            let idx = self.buffers.len();
            self.buffers.insert((type_id, raw_data.clone()), idx);
        }

        // Add the actual data bytes to the buffer.
        let offset = self.buffer.len();
        let size = std::mem::size_of_val(data);

        self.buffer.extend_from_slice(&raw_data);

        let (min, max) = if compute_bounds {
            (
                data.iter()
                    .copied()
                    .reduce(BufferValue::min)
                    .map(|e| BufferValue::value(&e)),
                data.iter()
                    .copied()
                    .reduce(BufferValue::max)
                    .map(|e| BufferValue::value(&e)),
            )
        } else {
            (None, None)
        };

        // Write a buffer view for the data.
        self.buffer_views.push(json::buffer::View {
            buffer: json::Index::new(0),
            byte_length: USize64::from(size),
            byte_offset: Some(USize64::from(offset)),
            byte_stride: None,
            target: target.map(Checked::Valid),
            name: None,
            extensions: Default::default(),
            extras: Default::default(),
        });

        // Write an accessor for the buffer view.
        self.accessors.push(json::Accessor {
            buffer_view: Some(json::Index::new(self.buffer_views.len() as u32 - 1)),
            byte_offset: None,
            component_type: Checked::Valid(GenericComponentType(E::COMPONENT_TYPE)),
            count: USize64::from(data.len()),
            min,
            max,
            normalized: false,
            type_: Checked::Valid(E::TYPE),
            sparse: None,
            name: None,
            extensions: Default::default(),
            extras: Default::default(),
        });

        self.accessors.len() - 1
    }

    pub fn push_node(&mut self, name: &str, node: &NodeData, parent: Option<usize>) {
        let mut output = json::Node::default();
        let idx = self.nodes.len();
        assert!(
            parent.map_or(true, |p| p < idx),
            "Parent nodes must be processed before children"
        );

        output.name = Some(name.to_string());

        // Transformation.
        if let Some(mtx) = &node.transform_matrix {
            output.matrix = Some(mtx.to_cols_array());
        } else if let Some(trs) = &node.transform {
            output.translation = Some(trs.translation.into());
            output.rotation = Some(UnitQuaternion(trs.rotation.to_array()));
            output.scale = Some(trs.scale.into());
        }

        // Reconstruct hierarchy.
        if let Some(parent) = parent {
            match self.nodes[parent].children {
                None => self.nodes[parent].children = Some(vec![json::Index::new(idx as u32)]),
                Some(ref mut children) => children.push(json::Index::new(idx as u32)),
            }
        }

        // Skin
        if let Some(skin) = &node.skin {
            let idx = self.push_skin(skin);
            output.skin = Some(json::Index::new(idx as u32));
        }

        // Mesh
        if let Some(mesh) = node.mesh.as_ref() {
            let idx = self.push_mesh(mesh);
            output.mesh = Some(json::Index::new(idx as u32));
        }

        // Animations
        for (name, animation) in &node.animations {
            self.add_anim_channels(idx, name, animation);
        }

        self.nodes.push(output);
    }

    fn push_mesh(&mut self, mesh: &Primitive) -> usize {
        let prim = self.make_primitive(mesh);
        self.meshes.push(json::Mesh {
            primitives: vec![prim],

            extensions: None,
            extras: Default::default(),
            name: None,
            weights: None,
        });
        self.meshes.len() - 1
    }

    fn make_primitive(&mut self, p: &Primitive) -> json::mesh::Primitive {
        // Remove the IDM-nicety clumping.
        let indices = p.indices.iter().flatten().copied().collect::<Vec<_>>();
        let indices = self.push_index_data(&indices);

        let mut attributes = BTreeMap::new();

        if !p.positions.is_empty() {
            attributes.insert(
                Checked::Valid(json::mesh::Semantic::Positions),
                json::Index::new(self.push_position_data(&p.positions) as u32),
            );
        }

        if !p.normals.is_empty() {
            attributes.insert(
                Checked::Valid(json::mesh::Semantic::Normals),
                json::Index::new(self.push_attribute_data(&p.normals) as u32),
            );
        }

        if !p.tex_coords.is_empty() {
            attributes.insert(
                Checked::Valid(json::mesh::Semantic::TexCoords(0)),
                json::Index::new(self.push_attribute_data(&p.tex_coords) as u32),
            );
        }

        if !p.joints.is_empty() {
            attributes.insert(
                Checked::Valid(json::mesh::Semantic::Joints(0)),
                json::Index::new(self.push_attribute_data(&p.joints) as u32),
            );
        }

        if !p.weights.is_empty() {
            attributes.insert(
                Checked::Valid(json::mesh::Semantic::Weights(0)),
                json::Index::new(self.push_attribute_data(&p.weights) as u32),
            );
        }

        let material = (!p.material.is_empty())
            .then(|| json::Index::new(self.push_material(&p.material) as u32));

        json::mesh::Primitive {
            indices: Some(json::Index::new(indices as u32)),
            attributes,

            extensions: None,
            extras: Default::default(),

            material,
            mode: Checked::Valid(json::mesh::Mode::Triangles),
            targets: None,
        }
    }

    fn push_material(&mut self, material: &Material) -> usize {
        // Try to reuse an existing material.

        // The material has textures defined which aren't found in our texture
        // list yet. This must be a new material so we just add it.
        if matches!(material.base_color_texture, Some(ref t) if !self.textures.contains_key(t))
            || matches!(
                material.metallic_roughness_texture,
                Some(ref t) if !self.textures.contains_key(t)
            )
        {
            return self.add_material(material);
        }

        for (i, m) in self.materials.iter().enumerate() {
            if m.pbr_metallic_roughness
                .base_color_texture
                .as_ref()
                .map(|t| t.index.value())
                == material
                    .base_color_texture
                    .as_ref()
                    .map(|t| *self.textures.get(t).unwrap())
                && m.pbr_metallic_roughness
                    .metallic_roughness_texture
                    .as_ref()
                    .map(|t| t.index.value())
                    == material
                        .metallic_roughness_texture
                        .as_ref()
                        .map(|t| *self.textures.get(t).unwrap())
            {
                return i;
            }
        }

        self.add_material(material)
    }

    fn add_material(&mut self, material: &Material) -> usize {
        // Insert textures if they aren't known yet.
        if let Some(ref t) = material.base_color_texture {
            if !self.textures.contains_key(t) {
                self.textures.insert(t.clone(), self.textures.len());
            }
        }

        if let Some(ref t) = material.metallic_roughness_texture {
            if !self.textures.contains_key(t) {
                self.textures.insert(t.clone(), self.textures.len());
            }
        }

        let base_color_texture =
            material
                .base_color_texture
                .as_ref()
                .map(|t| json::texture::Info {
                    index: json::Index::new(*self.textures.get(t).unwrap() as u32),
                    tex_coord: 0,
                    extensions: Default::default(),
                    extras: Default::default(),
                });

        let metallic_roughness_texture =
            material
                .metallic_roughness_texture
                .as_ref()
                .map(|t| json::texture::Info {
                    index: json::Index::new(*self.textures.get(t).unwrap() as u32),
                    tex_coord: 0,
                    extensions: Default::default(),
                    extras: Default::default(),
                });

        self.materials.push(json::Material {
            pbr_metallic_roughness: json::material::PbrMetallicRoughness {
                base_color_texture,
                metallic_roughness_texture,
                ..Default::default()
            },
            ..Default::default()
        });

        self.materials.len() - 1
    }

    fn push_skin(&mut self, skin: &Skin) -> usize {
        let inverse_bind_matrices = if skin.inverse_bind_matrices.is_empty() {
            None
        } else {
            Some(json::Index::new(
                self.push_data(&skin.inverse_bind_matrices) as u32,
            ))
        };

        self.skins.push(json::Skin {
            inverse_bind_matrices,
            joints: skin
                .joints
                .iter()
                .map(|j| {
                    json::Index::new(*self.names.get(j).unwrap_or_else(|| {
                        // XXX: We should raise an error if name lookup fails.
                        panic!("Skin: Joint {} not found", j);
                    }) as u32)
                })
                .collect(),
            skeleton: None,

            extensions: Default::default(),
            extras: Default::default(),
            name: None,
        });
        self.skins.len() - 1
    }

    fn add_anim_channels(&mut self, node_id: usize, name: &str, animation: &Animation) {
        // Find animation with the given name. If it doesn't exist, create it.
        let anim_idx = if let Some(idx) = self
            .animations
            .iter()
            .position(|a| a.name.as_ref().map_or(false, |n| n == name))
        {
            idx
        } else {
            self.animations.push(json::Animation {
                channels: Default::default(),
                samplers: Default::default(),
                name: Some(name.to_string()),
                extensions: Default::default(),
                extras: Default::default(),
            });
            self.animations.len() - 1
        };

        // (path_type, input, output)
        let mut channels = Vec::new();

        if !animation.translation.is_empty() {
            let (input, output): (Vec<f32>, Vec<Vec3>) =
                animation.translation.iter().copied().unzip();
            let (input, output) = (self.push_input_data(&input), self.push_data(&output));
            channels.push((json::animation::Property::Translation, input, output));
        }

        if !animation.rotation.is_empty() {
            let (input, output): (Vec<f32>, Vec<Quat>) = animation
                .rotation
                .iter()
                .copied()
                .map(|(v, a)| (v, Quat::from(a)))
                .unzip();
            let (input, output) = (self.push_input_data(&input), self.push_data(&output));
            channels.push((json::animation::Property::Rotation, input, output));
        }

        if !animation.scale.is_empty() {
            let (input, output): (Vec<f32>, Vec<Vec3>) = animation.scale.iter().copied().unzip();
            let (input, output) = (self.push_input_data(&input), self.push_data(&output));
            channels.push((json::animation::Property::Scale, input, output));
        }

        if !animation.weight.is_empty() {
            let (input, output): (Vec<f32>, Vec<f32>) = animation.weight.iter().copied().unzip();
            let (input, output) = (self.push_input_data(&input), self.push_data(&output));
            channels.push((json::animation::Property::MorphTargetWeights, input, output));
        }

        for (path_type, input, output) in channels {
            self.animations[anim_idx]
                .samplers
                .push(json::animation::Sampler {
                    input: json::Index::new(input as u32),
                    // XXX: Always use linear interpolation, the engine I use
                    // only supports that.
                    interpolation: Checked::Valid(json::animation::Interpolation::Linear),
                    output: json::Index::new(output as u32),
                    extensions: Default::default(),
                    extras: Default::default(),
                });

            let sampler = json::Index::new(self.animations[anim_idx].samplers.len() as u32 - 1);
            self.animations[anim_idx]
                .channels
                .push(json::animation::Channel {
                    sampler,
                    target: json::animation::Target {
                        node: json::Index::new(node_id as u32),
                        path: Checked::Valid(path_type),
                        extensions: Default::default(),
                        extras: Default::default(),
                    },
                    extensions: Default::default(),
                    extras: Default::default(),
                });
        }
    }
}

pub(crate) trait BufferValue: Copy + 'static {
    const COMPONENT_TYPE: ComponentType;
    const TYPE: Type;

    fn value(&self) -> json::Value;

    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
}

impl BufferValue for u16 {
    const COMPONENT_TYPE: ComponentType = ComponentType::U16;
    const TYPE: Type = Type::Scalar;

    fn value(&self) -> json::Value {
        json::Value::from(vec![*self])
    }

    fn min(self, other: Self) -> Self {
        std::cmp::Ord::min(self, other)
    }

    fn max(self, other: Self) -> Self {
        std::cmp::Ord::max(self, other)
    }
}

impl BufferValue for u32 {
    const COMPONENT_TYPE: ComponentType = ComponentType::U32;
    const TYPE: Type = Type::Scalar;

    fn value(&self) -> json::Value {
        json::Value::from(vec![*self])
    }

    fn min(self, other: Self) -> Self {
        std::cmp::Ord::min(self, other)
    }

    fn max(self, other: Self) -> Self {
        std::cmp::Ord::max(self, other)
    }
}

impl BufferValue for [u8; 4] {
    const COMPONENT_TYPE: ComponentType = ComponentType::U8;
    const TYPE: Type = Type::Vec4;

    fn value(&self) -> json::Value {
        json::Value::from(*self)
    }

    fn min(self, _other: Self) -> Self {
        unimplemented!()
    }

    fn max(self, _other: Self) -> Self {
        unimplemented!()
    }
}

impl BufferValue for f32 {
    const COMPONENT_TYPE: ComponentType = ComponentType::F32;
    const TYPE: Type = Type::Scalar;

    fn value(&self) -> json::Value {
        json::Value::from(vec![*self])
    }

    fn min(self, other: Self) -> Self {
        f32::min(self, other)
    }

    fn max(self, other: Self) -> Self {
        f32::max(self, other)
    }
}

impl BufferValue for Vec2 {
    const COMPONENT_TYPE: ComponentType = ComponentType::F32;
    const TYPE: Type = Type::Vec2;

    fn value(&self) -> json::Value {
        json::Value::from(self.to_array())
    }

    fn min(self, other: Self) -> Self {
        Vec2::min(self, other)
    }

    fn max(self, other: Self) -> Self {
        Vec2::max(self, other)
    }
}

impl BufferValue for Vec3 {
    const COMPONENT_TYPE: ComponentType = ComponentType::F32;
    const TYPE: Type = Type::Vec3;

    fn value(&self) -> json::Value {
        json::Value::from(self.to_array())
    }

    fn min(self, other: Self) -> Self {
        Vec3::min(self, other)
    }

    fn max(self, other: Self) -> Self {
        Vec3::max(self, other)
    }
}

impl BufferValue for Vec4 {
    const COMPONENT_TYPE: ComponentType = ComponentType::F32;
    const TYPE: Type = Type::Vec4;

    fn value(&self) -> json::Value {
        json::Value::from(self.to_array())
    }

    fn min(self, other: Self) -> Self {
        Vec4::min(self, other)
    }

    fn max(self, other: Self) -> Self {
        Vec4::max(self, other)
    }
}

impl BufferValue for Quat {
    const COMPONENT_TYPE: ComponentType = ComponentType::F32;
    const TYPE: Type = Type::Vec4;

    fn value(&self) -> json::Value {
        json::Value::from(self.to_array())
    }

    fn min(self, _other: Self) -> Self {
        unimplemented!()
    }

    fn max(self, _other: Self) -> Self {
        unimplemented!()
    }
}

impl BufferValue for Mat4 {
    const COMPONENT_TYPE: ComponentType = ComponentType::F32;
    const TYPE: Type = Type::Mat4;

    fn value(&self) -> json::Value {
        json::Value::from(self.to_cols_array())
    }

    // min/max doesn't really make sense for matrices.
    fn min(self, _other: Self) -> Self {
        unimplemented!()
    }

    fn max(self, _other: Self) -> Self {
        unimplemented!()
    }
}
