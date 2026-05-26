{
  description = "hierokeryx — entity extraction and resolution dev shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
      pkgsFor = system: nixpkgs.legacyPackages.${system};
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = pkgsFor system;

          # C++ / system runtime libs needed by ML wheels (torch, transformers,
          # sentence-transformers, gliner). The PyPI wheels are linked against
          # libstdc++/libgcc/libgomp which are not on NixOS's default loader path.
          mlRuntimeLibs = pkgs.lib.optionals pkgs.stdenv.isLinux [
            pkgs.stdenv.cc.cc.lib   # libstdc++, libgcc_s, libgomp
            pkgs.zlib               # libz (sentence-transformers, tokenizers)
          ];
        in
        {
          default = pkgs.mkShell {
            packages = [
              pkgs.python313
              pkgs.uv
              # NixOS can't run uv-installed ruff because it's a dynamically
              # linked standalone binary, not a Python C extension. The
              # nixpkgs build is patched for nix's loader.
              pkgs.ruff
            ] ++ mlRuntimeLibs;

            # uv will create .venv/ pointing at the nix-provided python313
            # rather than downloading its own interpreter.
            env = {
              UV_PYTHON_PREFERENCE = "only-system";
              UV_PYTHON = "${pkgs.python313}/bin/python3.13";
            } // pkgs.lib.optionalAttrs pkgs.stdenv.isLinux {
              LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath mlRuntimeLibs;
            };

            shellHook = ''
              # Local HuggingFace cache so model downloads stay inside the project
              export HF_HOME="$PWD/.models"

              echo "hierokeryx dev shell: $(python --version), uv $(uv --version | awk '{print $2}')"
            '';
          };
        });
    };
}
