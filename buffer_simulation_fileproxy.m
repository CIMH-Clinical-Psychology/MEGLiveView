%% Simulate data stream from recorded file

addpath C:/Users/simon.kern/fieldtrip-20250509
ft_defaults
cfg                 = [];
cfg.minblocksize    = 0.0;
cfg.maxblocksize    = 1.0;
cfg.channel         = 'all';
cfg.speed           = 1.0;


while 1
    cfg.source.dataset  = 'C:/Users/simon.kern/Nextcloud/ZI/2024.10 EndogenousOscillations/TDLM-Endogenous-Oscillations/data/DSMR113-filtered.fif';
    cfg.target.datafile = 'buffer://localhost:1972';
    
    ft_realtime_fileproxy(cfg);
end