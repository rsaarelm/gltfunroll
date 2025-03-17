# glTF Unroller

[glTF](https://www.khronos.org/gltf/) is a nice 3D exchange format, but it's written in a very much machine-first way.
This isn't nice if you want to edit your scene by hand in a text editor.
`gltfunroll` is a tool that opinionatedly converts a glTF into a tree-shaped [IDM](https://github.com/rsaarelm/idm) document.
You can edit the IDM, then convert it back to glTF.

Usage:

    gltfunroll model.gltf  # Unrolls model.gltf and .bin into model.idm
    gltfunroll model.idm   # Rolls model.idm into model.gltf and .bin

The tool backs up files it will overwrite with running numbers, `model.gltf.1`, `model.gltf.2` (if `.1` already exists and is different from `.2`), and so on.
If it's about to write contents exactly identical to the original file, it will not make a backup or touch the file.

This is very much "works on my machine".
There are plenty of clever things you can do with glTF files it doesn't cover.
The use case is working with animated models for a video game, one object per glTF file.
The tool assumes the glTF file has one scene with one root node that is the root of all the objects in the scene.
