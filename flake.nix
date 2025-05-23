{
  description = "Python bot for ChatGPT on Telegram";

  inputs = {
    nixpkgs.url = "nixpkgs";
    flake-parts.url = "flake-parts";
    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { flake-parts, ... }@inputs:
    let
      name = "chatgpt-telegram-bot";
      makePkg = import ./nix/pkg.nix;
      overlay = final: _: {
        ${name} = final.python3Packages.callPackage makePkg { };
      };

    in
    # flake-parts boilerplate
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [
        inputs.treefmt-nix.flakeModule
      ];

      flake = {
        overlays.default = overlay;
        nixosModules.default = import ./nix/service.nix;
      };

      systems = inputs.nixpkgs.lib.systems.flakeExposed;


      perSystem = { system, config, pkgs, ... }: {
        packages.default = config.legacyPackages.${name};
        packages.${name} = config.packages.default;
        legacyPackages = pkgs;

        _module.args.pkgs = import inputs.nixpkgs {
          inherit system;
          overlays = [ overlay ];
        };

        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            python3
            pdm
            ruff
          ];
        };

        treefmt = {
          programs.mypy = {
            enable = true;
            directories.".".extraPythonPackages = config.packages.default.propagatedBuildInputs;
          };
          programs.nixpkgs-fmt.enable = true;
        };
      };
    };
}
