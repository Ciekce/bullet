use bullet_lib::{
    inputs, lr, optimiser, outputs, wdl, Activation, LocalSettings, Loss, TrainerBuilder, TrainingSchedule,
};

fn main() {
    bullet_lib::set_cbcs(true);
    #[rustfmt::skip]
    let mut trainer1 = TrainerBuilder::default()
        .quantisations(&[255, 64])
        .optimiser(optimiser::AdamW)
        .input(inputs::ChessBucketsMirrored::new([
             0,  1,  2,  3,
             4,  5,  6,  7,
             8,  9, 10, 11,
            12, 13, 14, 15,
            16, 17, 18, 19,
            20, 21, 22, 23,
            24, 25, 26, 27,
            28, 29, 30, 31,
        ]))
        .output_buckets(outputs::MaterialCount::<8>)
        .feature_transformer(64)
        .activate(Activation::SCReLU)
        .add_layer(1)
        .build();

    #[rustfmt::skip]
    let mut trainer2 = TrainerBuilder::default()
        .quantisations(&[255, 64])
        .optimiser(optimiser::AdamW)
        .input(inputs::ChessBucketsMergedKingsMirrored::new([
             0,  1,  2,  3,
             4,  5,  6,  7,
             8,  9, 10, 11,
            12, 13, 14, 15,
            16, 17, 18, 19,
            20, 21, 22, 23,
            24, 25, 26, 27,
            28, 29, 30, 31,
        ]))
        .output_buckets(outputs::MaterialCount::<8>)
        .feature_transformer(64)
        .activate(Activation::SCReLU)
        .add_layer(1)
        .build();

    let mut schedule = TrainingSchedule {
        net_id: std::env::args().nth(1).unwrap().to_string(),
        eval_scale: 400.0,
        ft_regularisation: 0.0,
        batch_size: 16_384,
        batches_per_superbatch: 6104,
        start_superbatch: 1,
        end_superbatch: 100,
        wdl_scheduler: wdl::ConstantWDL { value: 0.4 },
        lr_scheduler: lr::Warmup {
            inner: lr::CosineDecayLR { initial_lr: 0.001, final_lr: 0.001 * f32::powi(0.3, 3), final_superbatch: 100 },
            warmup_batches: 200,
        },
        loss_function: Loss::SigmoidMPE(2.5),
        save_rate: 100,
        optimiser_settings: optimiser::AdamWParams {
            decay: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            min_weight: -1.98,
            max_weight: 1.98,
        },
    };

    let settings = LocalSettings {
        threads: 8,
        data_file_paths: vec!["C:/NNUE/chess/viri_11b_20240709_interleaved.bin"],
        test_set: None,
        output_directory: "checkpoints",
    };

    trainer1.run(&schedule, &settings);
    schedule.net_id = std::env::args().nth(2).unwrap().to_string();
    trainer2.run(&schedule, &settings);

    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ] {
        let eval1 = trainer1.eval(fen);
        let eval2 = trainer2.eval(fen);
        println!("FEN:  {fen}");
        println!("EVAL 1: {}", 400.0 * eval1);
        println!("EVAL 2: {}", 400.0 * eval2);
    }
}
