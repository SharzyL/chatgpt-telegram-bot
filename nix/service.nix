{ config, pkgs, lib, ... }:

with lib;
let
  cfg = config.services.chatgpt-telegram-bot;
in
{
  options.services.chatgpt-telegram-bot = {
    enable = mkEnableOption "ChatGPT Telegram bot service";

    package = lib.mkPackageOption pkgs "chatgpt-telegram-bot" { };

    configFile = mkOption { type = types.path; };

    envFile = mkOption { type = types.path; };
  };

  config = mkIf cfg.enable {
    systemd.services.chatgpt-telegram-bot = {
      description = "ChatGPT Telegram bot service";
      after = [ "network-online.target"  ];
      wants = [ "network-online.target"  ];
      wantedBy = [ "multi-user.target" ];
      serviceConfig = {
        ExecStart = "${cfg.package}/bin/chatgpt-telegram-bot -c ${cfg.configFile}";
        StateDirectory = "chatgpt-telegram-bot";
        WorkingDirectory = "%S/chatgpt-telegram-bot";
        Restart = "on-failure";
        DynamicUser = true;
        EnvironmentFile = cfg.envFile;
      };
    };
  };
}

