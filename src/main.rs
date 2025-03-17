use std::{
    collections::{BTreeMap, HashSet},
    path::{Path, PathBuf},
};

// TODO: Try filtering out animation paths that don't seem to change any
// values along them. (Maybe make a --filter-paths CLI option)

// TODO: Texture map names in IDM.

// TODO: Texture filter (smooth vs nearest-neighbor) in IDM

// TODO: Deduplicate buffers if you see repeats of the same data, eg. with
// animation inputs. Roller needs a hashmap with type tag & byte vector as
// key...

use anyhow::{bail, Result};
use clap::Parser;
use glam::{Quat, Vec2, Vec3, Vec4};
use gltf::animation::util::ReadOutputs;
use gltf_json::{
    self as json,
    accessor::{ComponentType, GenericComponentType, Type},
    scene::UnitQuaternion,
    validation::{Checked, USize64},
};
use serde::{Deserialize, Serialize};

const VERSION: Option<&str> = option_env!("CARGO_PKG_VERSION");

/// An opinionated tool that unwraps glTF models to a more readable and
/// editable format.
#[derive(Debug, Parser)]
struct Args {
    /// File to transform. If .glb or .gltf, unroll to IDM. If .idm, roll to
    /// .gltf.
    pub file: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();

    match args.file.extension().unwrap().to_str().unwrap() {
        "glb" | "gltf" => unroll(&args.file),
        "idm" => roll(&args.file),
        _ => bail!("Unknown file type"),
    }
}

/// Unroll a glTF file.
fn unroll(file: impl AsRef<Path>) -> Result<()> {
    let file = file.as_ref();

    let idm_file = file.with_extension("idm");

    // Initial glTF loading.

    let ctx = Unroller::new(file)?;
    let root = Node::new(&ctx, &ctx.root_node()?)?;

    // Serialize to IDM and write to disk.
    safe_save(idm_file, idm::to_string(&root)?.as_bytes())?;

    Ok(())
}

/// Roll our format back to glTF.
fn roll(file: impl AsRef<Path>) -> Result<()> {
    let file = file.as_ref();

    // Read file and deserialize to Node with IDM.
    let input: Node = idm::from_str(&std::fs::read_to_string(file)?)?;

    // Change extension to .gltf and .bin.
    let gltf_file = file.with_extension("gltf");
    let bin_file = file.with_extension("bin");

    // Our stuff is all in the tree of nodes, so we walk that and write glTF-y
    // stuff out.

    // Nodes don't contain their names and the root name isn't contained
    // anywhere inside the file. Instead, it's the name of the whole file.
    let root_name = file.file_stem().unwrap().to_string_lossy();

    // Push node tree into the roller.
    let mut roller = Roller::new(root_name.to_string(), &input);
    for (name, node, parent) in NodeIter::new(root_name, &input) {
        roller.push_node(&name, node, parent);
    }

    // Write to disk.
    safe_save(bin_file, &roller.buffer)?;
    safe_save(
        gltf_file,
        serde_json::to_string_pretty(&json::Root::from(roller))?.as_bytes(),
    )?;

    Ok(())
}

struct Unroller {
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

// Don't bother with scenes, just have one root node.

// Because the output is IDM, the contents of the node are wrapped in this
// struct that expresses the tree structure. Actual contents, other than names
// and children, are stored in NodeData. Serializing Nodes will produce nice
// IDM files.

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Node((NodeData,), BTreeMap<String, Node>);

impl Node {
    pub fn new(ctx: &Unroller, source: &gltf::Node) -> Result<Self> {
        // TODO: Support node cameras. Not high priority for model
        // work.
        let mut children = BTreeMap::new();
        for child in source.children() {
            let name = child.name().map_or_else(gensym, |n| n.to_string());
            children.insert(name, Self::new(ctx, &child)?);
        }

        let skin = source.skin().map(|s| Skin::new(ctx, &s)).transpose()?;

        // Flatten matrix/TRS transform into Node attribute level.
        let mut transform = None;
        let mut transform_matrix = None;
        match source.transform() {
            gltf::scene::Transform::Matrix { matrix } => {
                transform_matrix = Some(Mat4::from(matrix));
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

struct NodeIter<'a> {
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
struct NodeData {
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
enum Camera {
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
struct Trs {
    #[serde(skip_serializing_if = "Trs::empty_translation")]
    translation: Vec3,
    #[serde(skip_serializing_if = "Trs::empty_rotation")]
    rotation: Quat,
    #[serde(skip_serializing_if = "Trs::empty_scale")]
    scale: Vec3,
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
struct Skin {
    pub joints: Vec<String>,
    pub inverse_bind_matrices: Vec<Mat4>,
}

impl Skin {
    pub fn new(ctx: &Unroller, source: &gltf::Skin) -> Result<Self> {
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
struct Primitive {
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
    pub fn new(ctx: &Unroller, primitive: &gltf::Primitive) -> Result<Self> {
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
struct Material {
    pub texture_map: Option<String>,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Channel {
    pub interpolation: Interpolation,
    pub path: AnimPath,
}

impl Channel {
    pub fn new(ctx: &Unroller, channel: &gltf::animation::Channel) -> Result<Self> {
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
enum Interpolation {
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
enum AnimPath {
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

/// Context object for rebuilding a glTF file from a `Node` tree.
#[derive(Clone, Debug)]
struct Roller {
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

trait BufferValue: Copy + 'static {
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

/// Custom wrapper type for Mat4, mostly so that we get a nice IDM
/// serialization as four rows via the nested arrays serialization type.
#[derive(Copy, Clone, Debug, Default, Serialize, Deserialize)]
#[serde(from = "[[f32; 4]; 4]", into = "[[f32; 4]; 4]")]
struct Mat4(glam::Mat4);

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

/// Hacky unique name generator for missing name data.
fn gensym() -> String {
    static mut COUNTER: u32 = 0;
    unsafe {
        let counter = COUNTER;
        COUNTER += 1;
        format!("gensym-{counter}")
    }
}

fn safe_save(file: impl AsRef<Path>, data: impl AsRef<[u8]>) -> Result<()> {
    let file = file.as_ref();
    // If file already exists and is identical to data, do nothing.
    if file.exists() && std::fs::read(file)? == data.as_ref() {
        return Ok(());
    }

    // Otherwise make a smart backup of the original.
    backup(file)?;

    std::fs::write(file, data)?;
    Ok(())
}

/// Smart backup that makes consecutive backups of a file, but reuses the last
/// backup as long as the file does not change.
fn backup(file: impl AsRef<Path>) -> Result<()> {
    let file = file.as_ref();

    if !file.exists() {
        // No need to backup what ins't there.
        return Ok(());
    }

    // Find the highest backup number.
    let mut highest = 1;
    for entry in std::fs::read_dir(file.parent().unwrap())? {
        let entry = entry?;
        let path = entry.path();
        let Some(ext) = path.extension() else {
            continue;
        };
        if path.file_stem() != Some(file.as_os_str()) {
            continue;
        }

        let Ok(n) = ext.to_string_lossy().parse::<u32>() else {
            continue;
        };
        if n > highest {
            highest = n;
        }
    }

    let last_backup = PathBuf::from(format!("{}.{}", file.to_string_lossy(), highest));

    if last_backup.exists() && std::fs::read(file)? == std::fs::read(&last_backup)? {
        eprintln!(
            "{} is already backed up in {}",
            file.to_string_lossy(),
            last_backup.to_string_lossy()
        );
        return Ok(());
    }

    std::fs::copy(file, &last_backup)?;
    Ok(())
}
