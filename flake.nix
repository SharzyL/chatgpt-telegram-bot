{
  inputs = {
    nixpkgs.url = "nixpkgs";
    flake-utils.url = "flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }@inputs:
    flake-utils.lib.eachDefaultSystem
      (system:
      let
        pkgs = import nixpkgs { inherit system; };
        pyEnv = pkgs.python3.withPackages (ps: with ps; [
          telethon
          python-socks
          cryptg
          openai
        ]);
      in
        {
          legacyPackages = pkgs;
          devShell = pkgs.mkShell {
             buildInputs = [
               pyEnv
             ];
          };
        }
      )
    // { inherit inputs; };
}
