use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use clap::Parser;
use gltf_json::{self as json};
use roll::Roller;

mod roll;
mod unroll;

pub use unroll::{Angle, Animation, Camera, Mat4, Material, Node, NodeData, Primitive, Trs};
use unroll::{NodeIter, Unroller};

pub(crate) const VERSION: Option<&str> = option_env!("CARGO_PKG_VERSION");

/// An opinionated tool that unwraps glTF models to a more readable and
/// editable format.
#[derive(Debug, Parser)]
struct Args {
    /// File to transform. If .glb or .gltf, unroll to IDM. If .idm, roll to
    /// .gltf.
    pub file: PathBuf,

    /// Do a roundtrip transformation back to the original format. You usually
    /// want this from glTF to glTF, possibly with extra options.
    #[clap(long)]
    pub roundtrip: bool,

    /// Fuse a scene graph with multiple transformed and animated mesh nodes
    /// into a single root mesh with skeletal animation.
    #[clap(long)]
    pub skeletize: bool,
}

fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();

    match args.file.extension().unwrap().to_str().unwrap() {
        "glb" | "gltf" => {
            let mut node = unroll(&args.file)?;
            if args.skeletize {
                node.skeletize();
            }

            if args.roundtrip {
                // Save right back to glTF.
                roll_gltf(&args.file, &node)
            } else {
                // Save to IDM.
                let idm_file = args.file.with_extension("idm");
                safe_save(&idm_file, idm::to_string(&node)?.as_bytes())?;
                eprintln!("Unrolled to {}", idm_file.to_string_lossy());
                Ok(())
            }
        }
        "idm" => {
            let mut node = load_idm(&args.file)?;
            if args.skeletize {
                node.skeletize();
            }
            if args.roundtrip {
                panic!("Roundtrip is only supported when starting from glTF");
            }
            roll_gltf(&args.file, &node)
        }
        _ => bail!("Unknown file type"),
    }
}

fn unroll(file: impl AsRef<Path>) -> Result<Node> {
    let file = file.as_ref();

    // Initial glTF loading.
    let ctx = Unroller::new(file)?;
    Node::new(&ctx, &ctx.root_node()?)
}

fn load_idm(path: impl AsRef<Path>) -> Result<Node> {
    Ok(idm::from_str(&std::fs::read_to_string(path)?)?)
}

fn roll_gltf(path: impl AsRef<Path>, root: &Node) -> Result<()> {
    let path = path.as_ref();

    // Nodes don't contain their names and the root name isn't contained
    // anywhere inside the file. Instead, it's the name of the whole file.
    let root_name = path.file_stem().unwrap().to_string_lossy();

    let gltf_file = path.with_extension("gltf");
    let bin_file = path.with_extension("bin");

    // Our stuff is all in the tree of nodes, so we walk that and write glTF-y
    // stuff out.

    // Push node tree into the roller.
    let mut roller = Roller::new(root_name.to_string(), root);
    for (name, node, parent) in NodeIter::new(root_name, root) {
        roller.push_node(&name, node, parent);
    }

    // Write to disk.
    safe_save(bin_file, &roller.buffer)?;

    let mut json = serde_json::to_string_pretty(&json::Root::from(roller))?;
    if !json.ends_with('\n') {
        json.push('\n');
    }

    safe_save(&gltf_file, json.as_bytes())?;

    eprintln!("Rolled to {}", gltf_file.to_string_lossy());

    Ok(())
}

fn safe_save(file: impl AsRef<Path>, data: impl AsRef<[u8]>) -> Result<()> {
    let file = file.as_ref();
    // If file already exists and is identical to data, do nothing.
    if file.exists() && std::fs::read(file)? == data.as_ref() {
        log::info!(
            "save: {} already exists and is equal to new contents, doing nothing",
            file.to_string_lossy()
        );
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
    let file_name = file.to_string_lossy();

    if !file.exists() {
        log::info!("backup: No existing {file_name}, no need to do backups");
        return Ok(());
    }

    // Find the highest backup number.
    let mut highest = 0;
    let backup_prefix = format!("{file_name}.");
    for entry in std::fs::read_dir(file.parent().unwrap())? {
        let path = entry?.path();
        if !path.is_file() {
            continue;
        }

        let path_name = path.to_string_lossy();

        // Look for existing backup files. For our {file_name}, they are
        // {file_name}.1, {file_name}.2, and so on.
        if let Some(ext) = path_name.strip_prefix(&backup_prefix) {
            if let Ok(n) = ext.parse::<u32>() {
                log::info!("backup: Found backup file {path_name}");
                if n > highest {
                    highest = n;
                }
            }
        }
    }

    log::info!("backup: Last backup found is {file_name}.{highest}");

    let last_backup = PathBuf::from(format!("{}.{}", file.to_string_lossy(), highest));

    if last_backup.exists() && std::fs::read(file)? == std::fs::read(&last_backup)? {
        log::info!(
            "backup: {file_name} is already backed up in {}, doing nothing",
            last_backup.to_string_lossy()
        );
        return Ok(());
    }

    let backup_path = PathBuf::from(format!("{file_name}.{}", highest + 1));
    log::info!("backup: Saving backup to {}", backup_path.to_string_lossy());
    std::fs::copy(file, &backup_path)?;
    Ok(())
}
