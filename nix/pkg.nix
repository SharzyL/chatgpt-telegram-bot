{ buildPythonPackage
, lib
, telethon
, python-socks
, openai
, anthropic
, cryptg
, diskcache
, loguru
, mistune
, systemd-python
, hatchling
, pytestCheckHook
}:

buildPythonPackage {
  name = "chatgpt-telegram-bot";
  pyproject = true;
  nativeBuildInputs = [ hatchling ];

  src = with lib.fileset; toSource {
    root = ./..;
    fileset = fileFilter (file: file.name != "flake.nix" && file.name != "nix") ./..;
  };

  propagatedBuildInputs = [
    telethon
    python-socks
    cryptg
    diskcache
    openai
    anthropic
    loguru
    mistune
    systemd-python
  ];
  nativeCheckInputs = [ pytestCheckHook ];
}

