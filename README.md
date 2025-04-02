# glTF Unroller

[glTF](https://www.khronos.org/gltf/) is a nice 3D exchange format, but it's written in a very much machine-first way.
This isn't nice if you want to edit your scene by hand in a text editor.
`gltfunroll` is a tool that opinionatedly converts a glTF into a tree-shaped [IDM](https://github.com/rsaarelm/idm) document.
You can edit the IDM, then convert it back to glTF.

## Basic usage

    gltfunroll model.gltf  # Unrolls model.gltf and .bin into model.idm
    gltfunroll model.idm   # Rolls model.idm into model.gltf and .bin

You might be working with an engine that supports only skeletal animation, not scene graph based animations.
It can be easier to model animated objects as a collection of rigid scene graph nodes, so `gltfunroll` provides an option to convert an animated scene graph into a single node with skeletal animation.
You can either unroll an IDM, them skeletize when converting back to glTF, or skeletize a glTF model in-place:

    gltfunroll --skeletize scene.idm  # Convert to skeletal animation, save scene.gltf.
    gltfunroll --skeletize --in-place scene.gltf  # Ditto

## Implementation details

The tool backs up files it will overwrite with running numbers, `model.gltf.1`, `model.gltf.2` (if `.1` already exists and is different from `.2`), and so on.
If it's about to write contents exactly identical to the original file, it will not make a backup or touch the file.

The spec of the IDM format is that it's the IDM serialization of a single `struct Node` value that represents the top-level glTF node of a single implicit single-node scene.
The name of the root node is the same as the filename of the glTF file without extension.

This is very much "works on my machine".
There are plenty of clever things you can do with glTF files it doesn't cover.
The use case is working with animated models for a video game, one object per glTF file.
The tool assumes the glTF file has one scene with one root node that is the root of all the objects in the scene.

The tool isn't attempting to cover all of glTF, a primary purpose is to keep the IDM format simple and easy to work with.
If you're dealing with photorealistic industry-grade models, you're probably going to have a bad time trying to do things by hand-editing plaintext files in any case.
The practical purpose is working with video game models that must be usable with the [raylib](https://www.raylib.com/) engine.

## Utilities for working with IDM models

Angles use a variant type.
The glTF format uses quaternions internally, so an exported IDM file will always have `quat 0 0.707 0 0.707` style values for angles.
However, the type system also supports an Euler angles and degrees based variant.
When manually editing the file, you can replace the quaternion value with `euler 0 90 0` to represent the same rotation.
The rotation is applied in order Y (yaw) then X (pitch) then Z (roll).

You can pipe bits of the IDM file through the `quat2euler` Rust-script included in the repository to convert the `quat` angles there to `euler` angles.

If you're annoyed by the model data having noisy rounding error numbers like 0.999999998 or 0.44999999, you can pipe the IDM file through the `numclean` Rust-script to round these to the closest decimal.
It *probably* won't munge the data.
