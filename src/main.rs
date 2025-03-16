use std::{
    collections::{BTreeMap, HashSet},
    path::{Path, PathBuf},
};

use anyhow::{bail, Result};
use clap::Parser;
use glam::{Mat4, Quat, Vec2, Vec3, Vec4};
use gltf::animation::util::ReadOutputs;
use serde::{Deserialize, Serialize};

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

    // Do smart backups so we don't lose data.
    backup(&idm_file)?;

    // Initial glTF loading.

    let ctx = Context::new(file)?;

    let root = Node::new(&ctx, &ctx.root_node()?)?;

    // Serialize to IDM.
    std::fs::write(idm_file, idm::to_string(&root)?)?;

    Ok(())
}

/// Roll our format back to glTF.
fn roll(file: impl AsRef<Path>) -> Result<()> {
    let file = file.as_ref();

    // Read file and deserialize to Node with IDM.
    let _node: Node = idm::from_str(&std::fs::read_to_string(file)?)?;

    // Change extension to .gltf and .bin.
    let gltf_file = file.with_extension("gltf");
    let bin_file = file.with_extension("bin");

    // Do smart backups so we don't lose data.
    backup(&gltf_file)?;
    backup(&bin_file)?;

    todo!("Roll IDM back to glTF");
}

struct Context {
    pub gltf: gltf::Document,
    pub buffers: Vec<gltf::buffer::Data>,
}

impl Context {
    pub fn new(path: impl AsRef<Path>) -> Result<Context> {
        let (gltf, buffers, _images) = gltf::import(path)?;
        Ok(Context { gltf, buffers })
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

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Branch((Node,), BTreeMap<String, Branch>);

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
struct Node {
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub mesh: Vec<Primitive>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skin: Option<Skin>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transform: Option<Trs>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transform_matrix: Option<Mat4>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub camera: Option<Camera>,
    // Animation implicitly attached to one node.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub animations: BTreeMap<String, Vec<Channel>>,
}

impl Node {
    pub fn new(ctx: &Context, source: &gltf::Node) -> Result<Branch> {
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
                transform_matrix = Some(Mat4::from_cols_array_2d(&matrix));
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

        Ok(Branch(
            (Node {
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
struct Trs {
    translation: Vec3,
    rotation: Quat,
    scale: Vec3,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
struct Skin {
    pub joints: Vec<String>,
    pub inverse_bind_matrices: Vec<[[f32; 4]; 4]>,
}

impl Skin {
    pub fn new(ctx: &Context, source: &gltf::Skin) -> Result<Self> {
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
    pub fn new(ctx: &Context, primitive: &gltf::Primitive) -> Result<Self> {
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

        // TODO: Parse primitive data, needs buffer access
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
    pub fn new(ctx: &Context, channel: &gltf::animation::Channel) -> Result<Self> {
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
        // TODO: Parse outputs, figure out the variant stuff.
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

/// Hacky unique name generator for missing name data.
fn gensym() -> String {
    static mut COUNTER: u32 = 0;
    unsafe {
        let counter = COUNTER;
        COUNTER += 1;
        format!("gensym#{counter}")
    }
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
