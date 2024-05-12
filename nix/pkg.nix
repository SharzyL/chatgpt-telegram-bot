{ buildPythonPackage
, fetchFromGitHub
, lib
, telethon
, python-socks
, openai
, cryptg
, pdm-backend
}:

let
  telethon_1_32 = telethon.overridePythonAttrs (oldAttrs: rec {
    version = "1.32.1";
    src = fetchFromGitHub {
      owner = "LonamiWebs";
      repo = "Telethon";
      rev = "refs/tags/v${version}";
      hash = "sha256-0477SxYRVqRnCDPsu+q9zxejCnKVj+qa5DmH0VHuJyI=";
    };
    doCheck = false;
  });
in
buildPythonPackage {
  name = "chatgpt-telegram-bot";
  pyproject = true;
  nativeBuildInputs = [ pdm-backend ];

  src = with lib.fileset; toSource {
    root = ./..;
    fileset = fileFilter (file: file.name != "flake.nix" && file.name != "nix") ./..;
  };

  propagatedBuildInputs = [
    telethon_1_32
    python-socks
    cryptg
    openai
  ];
  doCheck = false; # since we have no test
}

