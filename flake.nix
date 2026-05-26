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

          # Python interpreter with pylsp bundled into its site-packages so
          # editors that invoke `python -m pylsp` (rather than the `pylsp`
          # binary directly) can find it. The .venv uv creates from this
          # interpreter does NOT inherit these — venv site-packages start
          # empty, so project deps stay isolated.
          python = pkgs.python313.withPackages (ps: [
            ps.python-lsp-server
          ]);

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
              python
              pkgs.uv
              # NixOS can't run uv-installed ruff because it's a dynamically
              # linked standalone binary, not a Python C extension. The
              # nixpkgs build is patched for nix's loader.
              pkgs.ruff
              # pylsp comes from `python.withPackages` above so both
              # `pylsp` and `python -m pylsp` resolve. For project deps
              # (numpy, pydantic, ...) to show up in completion, point
              # jedi at the venv in your editor config:
              #   pylsp.plugins.jedi.environment = ".venv/bin/python"
            ] ++ mlRuntimeLibs;

            # uv will create .venv/ pointing at the nix-provided python
            # rather than downloading its own interpreter.
            env = {
              UV_PYTHON_PREFERENCE = "only-system";
              UV_PYTHON = "${python}/bin/python3.13";
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
