use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use clap::Parser;
use gltf_json::{self as json};
use roll::Roller;
use serde::{Deserialize, Serialize};

mod roll;
mod unroll;

pub use unroll::{Animation, Camera, Material, Node, NodeData, Primitive, Skin, Trs};
use unroll::{NodeIter, Unroller};

pub(crate) const VERSION: Option<&str> = option_env!("CARGO_PKG_VERSION");

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
    safe_save(&idm_file, idm::to_string(&root)?.as_bytes())?;
    eprintln!("Unrolled to {}", idm_file.to_string_lossy());

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
        &gltf_file,
        serde_json::to_string_pretty(&json::Root::from(roller))?.as_bytes(),
    )?;

    eprintln!("Rolled to {}", gltf_file.to_string_lossy());

    Ok(())
}

/// Custom wrapper type for Mat4, mostly so that we get a nice IDM
/// serialization as four rows via the nested arrays serialization type.
#[derive(Copy, Clone, Debug, Default, Serialize, Deserialize)]
#[serde(from = "[[f32; 4]; 4]", into = "[[f32; 4]; 4]")]
pub struct Mat4(pub glam::Mat4);

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
