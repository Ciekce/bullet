use stoatformat::ShogiBoard;

use super::SparseInputType;

#[derive(Clone, Copy, Debug, Default)]
pub struct Shogi2344;
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
            let c = usize::from(piece & 1 > 0);
            if square < 81 {
                let pc = 81 * usize::from(piece >> 1);
                let sq = usize::from(square);

                let stm = [0, 1172][c] + pc + sq;
                let ntm = [1172, 0][c] + pc + (80 - sq);
                f(stm, ntm);
            } else { // hand
                let count = usize::from(square - 81);
                assert_ne!(count, 0);

                let pc = usize::from(piece >> 1);
                let offset = [0, 18, 22, 26, 30, 34, 36][pc];

                for i in 0..count {
                    let stm = [1134, 2306][c] + offset + i;
                    let ntm = [2306, 1134][c] + offset + i;
                    f(stm, ntm);
                }
            }
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
