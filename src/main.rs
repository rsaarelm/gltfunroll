use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use clap::Parser;
use gltf_json::{self as json};
use roll::Roller;

mod roll;
mod unroll;

use unroll::NodeIter;
pub use unroll::{Angle, Animation, Camera, Gltf, Mat4, Material, Node, NodeData, Primitive, Trs};

pub(crate) const VERSION: Option<&str> = option_env!("CARGO_PKG_VERSION");

/// An opinionated tool that unwraps glTF models to a more readable and
/// editable format.
#[derive(Debug, Parser)]
struct Args {
    /// Input file to transform. If it's .glb or .gltf, it's read as a glTF
    /// file, if it's .idm, it's read as an IDM file.
    pub input: PathBuf,

    /// Output file, extension must be .gltf or .idm.
    pub output: PathBuf,

    /// Fuse a scene graph with multiple transformed and animated mesh nodes
    /// into a single root mesh with skeletal animation.
    #[clap(long)]
    pub skeletize: bool,

    /// Merge all animations into a single named animation.
    ///
    /// Blender often produces files with different animation names for every
    /// sub-object, while the combined mesh should have a single animation
    /// name.
    #[clap(long, name = "NAME")]
    pub merge_animations: Option<String>,
}

fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();

    // Read input into IDM file.
    let mut gltf = match args.input.extension().unwrap().to_str().unwrap() {
        // If it's glTF, unroll it first.
        "glb" | "gltf" => Gltf::new(&args.input)?,
        "idm" => idm::from_str(&std::fs::read_to_string(&args.input)?)?,
        _ => bail!("Unknown input file type"),
    };

    if let Some(name) = args.merge_animations {
        gltf.merge_animations(&name);
    }

    // Turn a scene graph into a single mesh with skeletal animation.
    if args.skeletize {
        gltf.skeletize()?;
    }

    // Write output.
    match args.output.extension().unwrap().to_str().unwrap() {
        "gltf" => roll_gltf(&args.output, &gltf),
        "idm" => {
            let idm = idm::to_string(&gltf)?;
            std::fs::write(&args.output, idm.as_bytes())?;
            eprintln!("Unrolled to {}", args.output.to_string_lossy());
            Ok(())
        }
        _ => bail!("Unknown output file type"),
    }
}

fn roll_gltf(path: impl AsRef<Path>, gltf: &Gltf) -> Result<()> {
    let path = path.as_ref();

    // Nodes don't contain their names and the root name isn't contained
    // anywhere inside the file. Instead, it's the name of the whole file.
    let filename = path.file_stem().unwrap().to_string_lossy();

    let gltf_file = path.with_extension("gltf");
    let bin_file = path.with_extension("bin");

    // Our stuff is all in the tree of nodes, so we walk that and write glTF-y
    // stuff out.

    // Push node tree into the roller.
    let mut roller = Roller::new(filename, gltf);
    for (root_name, root) in &gltf.nodes {
        for (name, node, parent) in NodeIter::new(root_name, root) {
            roller.push_node(&name, node, parent);
        }
    }

    for (anim_name, a) in &gltf.animations {
        for (node_name, anim) in a {
            roller.push_anim(anim_name, node_name, anim);
        }
    }

    // Write to disk.
    std::fs::write(bin_file, &roller.buffer)?;

    let mut json = serde_json::to_string_pretty(&json::Root::from(roller))?;
    if !json.ends_with('\n') {
        json.push('\n');
    }

    std::fs::write(&gltf_file, json.as_bytes())?;

    eprintln!("Rolled to {}", gltf_file.to_string_lossy());

    Ok(())
}
