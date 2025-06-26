use bullet_lib::{
    game::{
        formats::stoatformat::{
            shogi::{position::Position, shogimove::Move},
            Outcome,
        },
        inputs::*,
        outputs,
    },
    nn::{optimiser, InitSettings},
    trainer::{
        logger,
        save::SavedFormat,
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
        NetworkTrainer,
    },
    value::{loader, ValueTrainerBuilder},
    Shape,
};

fn main() {
    logger::set_cbcs(true);

    const SCALE: f32 = 400.0;

    const SUPERBATCHES: usize = 200;

    const L1_SIZE: usize = 1024;
    const L2_SIZE: usize = 16;
    const L3_SIZE: usize = 32;

    const INPUT_BUCKETS: usize = 1;
    const OUTPUT_BUCKETS: usize = 1;

    let inputs = Shogi2344Mirrored;
    //let outputs = outputs::Single;

    let save_format = [
        //SavedFormat::id("ftf"), // factoriser
        SavedFormat::id("ftw"),
        SavedFormat::id("ftb"),
        SavedFormat::id("l1w"),
        SavedFormat::id("l1b"),
        SavedFormat::id("l2w"),
        SavedFormat::id("l2b"),
        SavedFormat::id("l3w"),
        SavedFormat::id("l3b"),
    ];

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .inputs(inputs)
        //.output_buckets(outputs)
        .optimiser(optimiser::Ranger)
        .save_format(&save_format)
        .loss_fn(|output, targets| output.sigmoid().squared_error(targets))
        .build(|builder, stm, ntm| {
            let mut ft = builder.new_affine("ft", inputs.num_inputs(), L1_SIZE);

            //let ftf = builder.new_weights("ftf", Shape::new(L1_SIZE, 2344), InitSettings::Zeroed);
            //let expanded = ftf.repeat(INPUT_BUCKETS);

            //ft.weights = ft.weights + expanded;

            let l1 = builder.new_affine("l1", L1_SIZE, OUTPUT_BUCKETS * L2_SIZE);
            let l2 = builder.new_affine("l2", L2_SIZE * 2, OUTPUT_BUCKETS * L3_SIZE);
            let l3 = builder.new_affine("l3", L3_SIZE, OUTPUT_BUCKETS);

            let stm_subnet = ft.forward(stm).crelu().pairwise_mul();
            let ntm_subnet = ft.forward(ntm).crelu().pairwise_mul();
            let mut out = stm_subnet.concat(ntm_subnet);

            out = l1.forward(out);
            out = out.concat(out.abs_pow(2.0)).crelu();

            out = l2.forward(out).crelu();
            out = l3.forward(out);

            out
        });

    let net_id = std::env::args().nth(1).unwrap();

    let schedule = TrainingSchedule {
        net_id: net_id.to_string(),
        eval_scale: SCALE,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: SUPERBATCHES,
        },
        wdl_scheduler: wdl::LinearWDL { start: 0.1, end: 0.4 },
        lr_scheduler: lr::CosineDecayLR {
            initial_lr: 0.001,
            final_lr: 0.000027,
            final_superbatch: SUPERBATCHES,
        },
        save_rate: SUPERBATCHES,
    };

    let default_optimiser_params = optimiser::RangerParams { min_weight: -1.98, max_weight: 1.98, ..Default::default() };

    /*
    let ftw_optimiser_params = optimiser::RangerParams {
        min_weight: -0.99,
        max_weight: 0.99,
        ..default_optimiser_params
    };
    */

    let l1w_clip = 0.99 * 255.0 * 255.0 / (256.0 * 256.0);

    let l1w_optimiser_params =
        optimiser::RangerParams { min_weight: -l1w_clip, max_weight: l1w_clip, ..default_optimiser_params };

    trainer.optimiser.set_params(default_optimiser_params);

    //trainer.optimiser.set_params_for_weight("ftw", ftw_optimiser_params);
    //trainer.optimiser.set_params_for_weight("ftf", ftw_optimiser_params); // factoriser

    trainer.optimiser.set_params_for_weight("l1w", l1w_optimiser_params);

    let settings = LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 64 };

    let data_loader = {
        let file_path = "data.spk";
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
