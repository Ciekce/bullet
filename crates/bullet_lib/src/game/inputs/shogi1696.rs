use stoatformat::ShogiBoard;

use super::SparseInputType;

#[derive(Clone, Copy, Debug, Default)]
pub struct Shogi1696;

impl Shogi1696 {
    fn map_psqt_feature(piece: u8, stm_sq: u8, ntm_sq: u8) -> (usize, usize) {
        const PIECE_MAP: [usize; 14] = [0, 4, 1, 2, 4, 4, 3, 4, 4, 5, 6, 7, 8, 9];

        let c = usize::from(piece & 1 > 0);
        let pc = 81 * PIECE_MAP[(piece >> 1) as usize];

        let stm_sq = usize::from(stm_sq);
        let ntm_sq = usize::from(ntm_sq);

        let stm = [0, 848][c] + pc + stm_sq;
        let ntm = [848, 0][c] + pc + ntm_sq;
        println!("{}, {}", stm, ntm);
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
        let stm = [809, 1657][c] + offset + count;
        let ntm = [1657, 809][c] + offset + count;
        (stm, ntm)
    }
}

impl SparseInputType for Shogi1696 {
    type RequiredDataType = ShogiBoard;

    /// The total number of inputs
    fn num_inputs(&self) -> usize {
        1696
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
        "1696".to_string()
    }

    /// Description of the input type
    fn description(&self) -> String {
        "Default psqt+hand shogi inputs, with merged golds".to_string()
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Shogi1696Mirrored;
impl SparseInputType for Shogi1696Mirrored {
    type RequiredDataType = ShogiBoard;

    /// The total number of inputs
    fn num_inputs(&self) -> usize {
        1696
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
                Shogi1696::map_psqt_feature(piece, stm_sq, ntm_sq)
            } else {
                Shogi1696::map_hand_feature(piece, square - 81)
            };
            f(stm, ntm);
        }
    }

    /// Shorthand for the input e.g. `768x4`
    fn shorthand(&self) -> String {
        "1696hm".to_string()
    }

    /// Description of the input type
    fn description(&self) -> String {
        "Default psqt+hand shogi inputs, with merged golds, horizontally mirrored".to_string()
    }
}
