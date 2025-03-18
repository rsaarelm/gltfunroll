use std::collections::BTreeMap;

use glam::{Quat, Vec2, Vec3, Vec4};
use gltf_json::{
    self as json,
    accessor::{ComponentType, GenericComponentType, Type},
    scene::UnitQuaternion,
    validation::{Checked, USize64},
};

use crate::{unroll::NodeIter, AnimPath, Channel, Mat4, Node, NodeData, Primitive, Skin, VERSION};

/// Context object for rebuilding a glTF file from a `Node` tree.
#[derive(Clone, Debug)]
pub(crate) struct Roller {
    pub names: BTreeMap<String, usize>,

    pub accessors: Vec<json::Accessor>,
    pub animations: Vec<json::Animation>,
    pub buffer: Vec<u8>,
    pub buffer_views: Vec<json::buffer::View>,
    pub meshes: Vec<json::Mesh>,
    pub nodes: Vec<json::Node>,
    pub skins: Vec<json::Skin>,
}

impl From<Roller> for json::Root {
    fn from(ctx: Roller) -> Self {
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
                uri: None,
                name: None,
                extensions: Default::default(),
                extras: Default::default(),
            }],
            cameras: Default::default(),

            extensions: Default::default(),
            extensions_used: Default::default(),
            extensions_required: Default::default(),
            extras: Default::default(),

            // TODO Roll in texture map names
            images: Default::default(),

            // TODO Roll in materials
            materials: Default::default(),
            meshes: ctx.meshes,
            nodes: ctx.nodes,
            // TODO Roll in texture samplers
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
            // TODO Roll in textures
            textures: Default::default(),
        }
    }
}

impl Roller {
    /// Build the roller, initialize name lookup for the node tree.
    pub fn new(root_name: impl Into<String>, root: &Node) -> Self {
        let mut names = BTreeMap::new();
        for (i, (name, _, _)) in NodeIter::new(root_name.into(), root).enumerate() {
            names.insert(name, i);
        }

        Self {
            names,

            accessors: Default::default(),
            animations: Default::default(),
            buffer: Default::default(),
            buffer_views: Default::default(),
            meshes: Default::default(),
            nodes: Default::default(),
            skins: Default::default(),
        }
    }

    /// Push a buffer of data into the output glTF.
    ///
    /// Return the accessor index for the data.
    pub fn push_data<E>(&mut self, data: &[E], compute_bounds: bool) -> usize
    where
        E: BufferValue,
    {
        // Add the actual data bytes to the buffer.
        let offset = self.buffer.len();
        let size = std::mem::size_of_val(data);

        self.buffer.extend_from_slice(unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, size)
        });

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
            target: None,
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
        if !node.mesh.is_empty() {
            let idx = self.push_mesh(&node.mesh);
            output.mesh = Some(json::Index::new(idx as u32));
        }

        // Animations
        for (name, channels) in &node.animations {
            self.add_anim_channels(idx, name, channels);
        }

        self.nodes.push(output);
    }

    fn push_mesh(&mut self, mesh: &[Primitive]) -> usize {
        let primitives = mesh.iter().map(|p| self.make_primitive(p)).collect();
        self.meshes.push(json::Mesh {
            primitives,

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
        let indices = self.push_data(&indices, false);

        let mut attributes = BTreeMap::new();

        if !p.positions.is_empty() {
            attributes.insert(
                Checked::Valid(json::mesh::Semantic::Positions),
                json::Index::new(self.push_data(&p.positions, true) as u32),
            );
        }

        if !p.normals.is_empty() {
            attributes.insert(
                Checked::Valid(json::mesh::Semantic::Normals),
                json::Index::new(self.push_data(&p.normals, false) as u32),
            );
        }

        if !p.tex_coords.is_empty() {
            attributes.insert(
                Checked::Valid(json::mesh::Semantic::TexCoords(0)),
                json::Index::new(self.push_data(&p.tex_coords, false) as u32),
            );
        }

        if !p.joints.is_empty() {
            attributes.insert(
                Checked::Valid(json::mesh::Semantic::Joints(0)),
                json::Index::new(self.push_data(&p.joints, false) as u32),
            );
        }

        if !p.weights.is_empty() {
            attributes.insert(
                Checked::Valid(json::mesh::Semantic::Weights(0)),
                json::Index::new(self.push_data(&p.weights, false) as u32),
            );
        }

        json::mesh::Primitive {
            indices: Some(json::Index::new(indices as u32)),
            attributes,

            extensions: None,
            extras: Default::default(),

            // TODO: Assign material to primitive.
            material: None,
            mode: Checked::Valid(json::mesh::Mode::Triangles),
            targets: None,
        }
    }

    fn push_skin(&mut self, skin: &Skin) -> usize {
        let inverse_bind_matrices = Some(json::Index::new(
            self.push_data(&skin.inverse_bind_matrices, false) as u32,
        ));

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

    fn add_anim_channels(&mut self, node_id: usize, name: &str, channels: &[Channel]) {
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

        for c in channels {
            // split anim path into type (the enum variant), input timestamps (first
            // items of value tuples) and output values (second items of value
            // tuples)
            let (path_type, input, output) = match &c.path {
                AnimPath::Translation(t) => {
                    let (input, output): (Vec<f32>, Vec<Vec3>) = t.iter().copied().unzip();
                    let (input, output) =
                        (self.push_data(&input, true), self.push_data(&output, false));
                    (json::animation::Property::Translation, input, output)
                }
                AnimPath::Rotation(t) => {
                    let (input, output): (Vec<f32>, Vec<Quat>) = t.iter().copied().unzip();
                    let (input, output) =
                        (self.push_data(&input, true), self.push_data(&output, false));
                    (json::animation::Property::Rotation, input, output)
                }
                AnimPath::Scale(t) => {
                    let (input, output): (Vec<f32>, Vec<Vec3>) = t.iter().copied().unzip();
                    let (input, output) =
                        (self.push_data(&input, true), self.push_data(&output, false));
                    (json::animation::Property::Scale, input, output)
                }
                AnimPath::Weights(t) => {
                    let (input, output): (Vec<f32>, Vec<f32>) = t.iter().copied().unzip();
                    let (input, output) =
                        (self.push_data(&input, true), self.push_data(&output, false));
                    (json::animation::Property::MorphTargetWeights, input, output)
                }
            };

            self.animations[anim_idx]
                .samplers
                .push(json::animation::Sampler {
                    input: json::Index::new(input as u32),
                    interpolation: Checked::Valid(c.interpolation.into()),
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

impl BufferValue for u32 {
    const COMPONENT_TYPE: ComponentType = ComponentType::U32;
    const TYPE: Type = Type::Scalar;

    fn value(&self) -> json::Value {
        json::Value::from(*self)
    }

    fn min(self, other: Self) -> Self {
        std::cmp::Ord::min(self, other)
    }

    fn max(self, other: Self) -> Self {
        std::cmp::Ord::max(self, other)
    }
}

impl BufferValue for [u16; 4] {
    const COMPONENT_TYPE: ComponentType = ComponentType::U16;
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
        json::Value::from(*self)
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
