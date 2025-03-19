use bullet_lib::{
    nn::{optimiser, Activation},
    trainer::{
        default::{
            formats::stoatformat::{
                shogi::{position::Position, shogimove::Move},
                Outcome,
            },
            inputs, loader, outputs, Loss, TrainerBuilder,
        },
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
};

fn main() {
    bullet_lib::logger::set_cbcs(true);

    const SCALE: f32 = 400.0;
    const SUPERBATCHES: usize = 180;

    let mut trainer = TrainerBuilder::default()
        .quantisations(&[255, 64])
        .optimiser(optimiser::AdamW)
        .loss_fn(Loss::SigmoidMSE)
        .input(inputs::Shogi2344Mirrored)
        .output_buckets(outputs::Single)
        .feature_transformer(256)
        .activate(Activation::SCReLU)
        .add_layer(1)
        .build();

    let schedule = TrainingSchedule {
        net_id: std::env::args().nth(1).unwrap().to_string(),
        eval_scale: SCALE,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: SUPERBATCHES,
        },
        wdl_scheduler: wdl::LinearWDL { start: 0.0, end: 0.3 },
        lr_scheduler: lr::CosineDecayLR {
            initial_lr: 0.001,
            final_lr: 0.001 * f32::powi(0.3, 3),
            final_superbatch: SUPERBATCHES,
        },
        save_rate: SUPERBATCHES,
    };

    trainer.set_optimiser_params(optimiser::AdamWParams::default());

    let settings = LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 64 };

    let data_loader = {
        let file_path = "data.bin";
        let buffer_size_mb = 8192;
        let threads = 4;
        fn filter(pos: &Position, mv: Move, score: i16, _wdl: Outcome) -> bool {
            true
                //&& pos.ply_count() > 40
                && !pos.is_capture(mv)
                && score.unsigned_abs() < 25000
                && !pos.is_in_check()
        }

        loader::StoatpackLoader::new(file_path, buffer_size_mb, threads, filter)
    };

    trainer.run(&schedule, &settings, &data_loader);

    for sfen in [
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
        "8l/1l+R2P3/p2pBG1pp/kps1p4/Nn1P2G2/P1P1P2PP/1PS6/1KSG3+r1/LN2+p3L w Sbgn3p 124",
        "lnsgkgsnl/1r7/p1ppp1bpp/1p3pp2/7P1/2P6/PP1PPPP1P/1B3S1R1/LNSGKG1NL b - 9",
        "l4S2l/4g1gs1/5p1p1/pr2N1pkp/4Gn3/PP3PPPP/2GPP4/1K7/L3r+s2L w BS2N5Pb 1",
        "6n1l/2+S1k4/2lp4p/1np1B2b1/3PP4/1N1S3rP/1P2+pPP+p1/1p1G5/3KG2r1 b GSN2L4Pgs2p 1",
        "l6nl/5+P1gk/2np1S3/p1p4Pp/3P2Sp1/1PPb2P1P/P5GS1/R8/LN4bKL w RGgsn5p 1",
    ] {
        let eval = trainer.eval(sfen);
        println!("SFEN: {sfen}");
        println!("EVAL: {}", SCALE * eval);
    }
}
