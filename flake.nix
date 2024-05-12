{
  inputs = {
    nixpkgs.url = "nixpkgs";
    flake-utils.url = "flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }@inputs:
    let
      overlay = final: prev: {
        chatgpt-telegram-bot = final.python3.pkgs.callPackage ./nix/pkg.nix { };
      };
    in
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          pkgs = import nixpkgs { inherit system; overlays = [ overlay ]; };
          chatgpt-telegram-bot = pkgs.chatgpt-telegram-bot;
        in
        {
          packages.default = chatgpt-telegram-bot;
          legacyPackages = pkgs;
          devShell = chatgpt-telegram-bot.overrideAttrs (oldAttrs: {
            nativeBuildInputs = oldAttrs.nativeBuildInputs ++ [ pkgs.pdm ];
          });
        }
      )
    // {
      inherit inputs;
      overlays.default = overlay;
      nixosModules.default = {
        imports = [ ./nix/service.nix ];
      };
    };
}
