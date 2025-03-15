use std::{
    collections::{BTreeMap, HashSet},
    path::{Path, PathBuf},
};

use anyhow::{bail, Result};
use clap::Parser;
use glam::{Mat4, Vec2, Vec3, Vec4};
use is_terminal::IsTerminal;
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

    // Output an IDM file next to the glTF. If one already exists, ask if it's
    // okay to overwrite it. Except if we're not in an interactive session, in
    // which case overwriter goes brrr.
    let idm_file = file.with_extension("idm");
    if std::io::stdout().is_terminal() && idm_file.exists() {
        let ok = dialoguer::Confirm::new()
            .with_prompt(format!(
                "Overwrite existing {}?",
                idm_file.to_string_lossy()
            ))
            .interact()?;

        // If the user says no, bail.
        if !ok {
            eprintln!("Aborted.");
            return Ok(());
        }
    }

    // Initial glTF loading.

    let (gltf, _buffers, _images) = gltf::import(file)?;

    // Find the unique root node in gltf.
    let children: HashSet<usize> = gltf
        .nodes()
        .flat_map(|n| n.children().map(|n| n.index()))
        .collect();
    let mut root_nodes = gltf.nodes().filter(|n| !children.contains(&n.index()));
    let Some(root_node) = root_nodes.next() else {
        bail!("No root node found")
    };
    if root_nodes.next().is_some() {
        bail!("Multiple root nodes found");
    }

    let node = Node::new(&gltf, &root_node)?;

    // Serialize to IDM.
    std::fs::write(idm_file, idm::to_string(&node)?)?;

    Ok(())
}

/// Roll our format back to glTF.
fn roll(file: impl AsRef<Path>) -> Result<()> {
    let file = file.as_ref();

    // Read file and deserialize to Node with IDM.
    let _node: Node = idm::from_str(&std::fs::read_to_string(file)?)?;

    // Change extension to .gltf and .bin.
    let _gltf_file = file.with_extension("gltf");
    let _bin_file = file.with_extension("bin");

    todo!("Roll IDM back to glTF");
}

// Don't bother with scenes, just have one root node.

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
struct Node {
    children: BTreeMap<String, Node>,
    mesh: Vec<Primitive>,
    skin: Option<Skin>,
    camera: Option<Camera>,
    // Animation implicitly attached to one node.
    animations: BTreeMap<String, Vec<Channel>>,
}

impl Node {
    pub fn new(doc: &gltf::Document, source: &gltf::Node) -> Result<Self> {
        // TODO: Support node transforms. (Raylib doesn't like them so they're
        // not high priority.)

        // TODO: Support node cameras. Again, not high priority for model
        // work.
        let mut children = BTreeMap::new();
        for child in source.children() {
            let name = child.name().map_or_else(gensym, |n| n.to_string());
            children.insert(name, Self::new(doc, &child)?);
        }

        let skin = source.skin().map(|s| Skin::new(&s)).transpose()?;

        // XXX: Ignoring mesh weights. Do we need it?

        let mesh = source
            .mesh()
            .map(|m| {
                m.primitives()
                    .map(|p| Primitive::new(&p))
                    .collect::<Result<_>>()
            })
            .transpose()?
            .unwrap_or_default();

        let mut animations = BTreeMap::new();
        for a in doc.animations() {
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
                .map(|c| Channel::new(&c))
                .collect::<Result<_>>()?;
            animations.insert(name, channels);
        }

        Ok(Node {
            children,
            mesh,
            skin,
            animations,
            ..Default::default()
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
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

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Skin {
    joints: Vec<usize>,
    inverse_bind_matrices: Vec<Mat4>,
}

impl Skin {
    pub fn new(_source: &gltf::Skin) -> Result<Self> {
        // TODO: Parse skin data, needs buffer access
        Ok(Default::default())
        /*
        let joints = source.joints().map(|n| n.index()).collect();
        // How do get matrix data?
        Ok(Self {
            joints,
            inverse_bind_matrices,
        })
        */
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Primitive {
    attributes: Vec<Attribute>,
    indices: Vec<usize>,
    material: Material,
}

impl Primitive {
    pub fn new(_source: &gltf::Primitive) -> Result<Self> {
        // TODO: Parse primitive data, needs buffer access
        Ok(Default::default())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
enum Attribute {
    Position(Vec<Vec3>),
    Normal(Vec<Vec3>),
    TexCoord(Vec<Vec2>),
    Joints(Vec<Vec4>),
    Weights(Vec<Vec4>),
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Material {
    texture_map: Option<String>,
    metallic_factor: f32,
    roughness_factor: f32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Channel {
    input: Vec<f32>,
    output: Vec<f32>,
    interpolation: Interpolation,
    path: AnimPath,
}

impl Channel {
    pub fn new(_source: &gltf::animation::Channel) -> Result<Self> {
        // TODO: Parse channel data, needs buffer access
        Ok(Default::default())
    }
}

#[derive(Copy, Clone, Debug, Default, Serialize, Deserialize)]
enum Interpolation {
    #[default]
    Linear,
    Step,
    CubicSpline,
}

#[derive(Copy, Clone, Debug, Default, Serialize, Deserialize)]
enum AnimPath {
    #[default]
    Translation,
    Rotation,
    Scale,
    Weights,
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
