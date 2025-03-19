use stoatformat::ShogiBoard;

use super::SparseInputType;

#[derive(Clone, Copy, Debug, Default)]
pub struct Shogi2344;

impl Shogi2344 {
    fn map_psqt_feature(piece: u8, stm_sq: u8, ntm_sq: u8) -> (usize, usize) {
        let c = usize::from(piece & 1 > 0);
        let pc = 81 * usize::from(piece >> 1);

        let stm_sq = usize::from(stm_sq);
        let ntm_sq = usize::from(ntm_sq);

        let stm = [0, 1172][c] + pc + stm_sq;
        let ntm = [1172, 0][c] + pc + ntm_sq;
        (stm, ntm)
    }

    fn map_hand_feature(piece: u8, count: u8) -> (usize, usize) {
        let c = usize::from(piece & 1 > 0);

        let count = usize::from(count);
        assert_ne!(count, 0);

        // plnsgbr
        let pc = usize::from(piece >> 1);
        let offset = [0, 18, 22, 26, 30, 34, 36][pc];

        // -1 because count is 1-based
        let stm = [1133, 2305][c] + offset + count;
        let ntm = [2305, 1133][c] + offset + count;
        (stm, ntm)
    }
}

impl SparseInputType for Shogi2344 {
    type RequiredDataType = ShogiBoard;

    /// The total number of inputs
    fn num_inputs(&self) -> usize {
        2344
    }

    /// The maximum number of active inputs
    fn max_active(&self) -> usize {
        40
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        for (piece, square) in pos.into_iter() {
            let (stm, ntm) = if square < 81 {
                Self::map_psqt_feature(piece, square, 80 - square)
            } else {
                Self::map_hand_feature(piece, square - 81)
            };
            f(stm, ntm);
        }
    }

    /// Shorthand for the input e.g. `768x4`
    fn shorthand(&self) -> String {
        "2344".to_string()
    }

    /// Description of the input type
    fn description(&self) -> String {
        "Default psqt+hand shogi inputs".to_string()
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Shogi2344Mirrored;
impl SparseInputType for Shogi2344Mirrored {
    type RequiredDataType = ShogiBoard;

    /// The total number of inputs
    fn num_inputs(&self) -> usize {
        2344
    }

    /// The maximum number of active inputs
    fn max_active(&self) -> usize {
        40
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        fn flip_sq(sq: u8) -> u8 {
            let rank = sq / 9;
            let file = sq % 9;
            rank * 9 + (8 - file)
        }

        let flip_stm = pos.stm_king_sq() % 9 > 4;
        let flip_ntm = pos.nstm_king_sq() % 9 > 4;

        for (piece, square) in pos.into_iter() {
            let (stm, ntm) = if square < 81 {
                let stm_sq = if flip_stm { flip_sq(square) } else { square };
                let ntm_sq = if flip_ntm { flip_sq(80 - square) } else { 80 - square };
                Shogi2344::map_psqt_feature(piece, stm_sq, ntm_sq)
            } else {
                Shogi2344::map_hand_feature(piece, square - 81)
            };
            f(stm, ntm);
        }
    }

    /// Shorthand for the input e.g. `768x4`
    fn shorthand(&self) -> String {
        "2344hm".to_string()
    }

    /// Description of the input type
    fn description(&self) -> String {
        "Default psqt+hand shogi inputs, horizontally mirrored".to_string()
    }
}
