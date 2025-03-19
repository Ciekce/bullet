use stoatformat::ShogiBoard;

use super::{get_num_buckets, Factorises, Shogi2344, Shogi2344Mirrored, SparseInputType};

#[derive(Clone, Copy, Debug)]
pub struct ShogiBuckets {
    buckets: [usize; 81],
    num_buckets: usize,
}

impl ShogiBuckets {
    pub fn new(buckets: [usize; 81]) -> Self {
        Self { buckets, num_buckets: get_num_buckets(&buckets) }
    }
}

impl SparseInputType for ShogiBuckets {
    type RequiredDataType = ShogiBoard;

    /// The total number of inputs
    fn num_inputs(&self) -> usize {
        2344 * self.num_buckets
    }

    /// The maximum number of active inputs
    fn max_active(&self) -> usize {
        40
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        let our_bucket = 2344 * self.buckets[usize::from(pos.stm_king_sq())];
        let opp_bucket = 2344 * self.buckets[usize::from(pos.nstm_king_sq())];

        Shogi2344.map_features(pos, |stm, ntm| f(our_bucket + stm, opp_bucket + ntm));
    }

    /// Shorthand for the input e.g. `768x4`
    fn shorthand(&self) -> String {
        format!("2344x{}", self.num_buckets)
    }

    /// Description of the input type
    fn description(&self) -> String {
        "King bucketed psqt+hand shogi inputs".to_string()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ShogiBucketsMirrored {
    buckets: [usize; 81],
    num_buckets: usize,
}

impl Default for ShogiBucketsMirrored {
    fn default() -> Self {
        Self { buckets: [0; 81], num_buckets: 1 }
    }
}

impl ShogiBucketsMirrored {
    pub fn new(buckets: [usize; 45]) -> Self {
        let num_buckets = get_num_buckets(&buckets);

        let mut expanded = [0; 81];
        for (idx, elem) in expanded.iter_mut().enumerate() {
            *elem = buckets[(idx / 9) * 5 + [0, 1, 2, 3, 4, 3, 2, 1, 0][idx % 9]];
        }

        Self { buckets: expanded, num_buckets }
    }
}

impl SparseInputType for ShogiBucketsMirrored {
    type RequiredDataType = ShogiBoard;

    /// The total number of inputs
    fn num_inputs(&self) -> usize {
        2344 * self.num_buckets
    }

    /// The maximum number of active inputs
    fn max_active(&self) -> usize {
        40
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        let our_bucket = 2344 * self.buckets[usize::from(pos.stm_king_sq())];
        let opp_bucket = 2344 * self.buckets[usize::from(pos.nstm_king_sq())];

        Shogi2344Mirrored.map_features(pos, |stm, ntm| f(our_bucket + stm, opp_bucket + ntm));
    }

    /// Shorthand for the input e.g. `768x4`
    fn shorthand(&self) -> String {
        format!("2344x{}hm", self.num_buckets)
    }

    /// Description of the input type
    fn description(&self) -> String {
        "Horizontally mirrored, king bucketed psqt+hand shogi inputs".to_string()
    }
}

impl Factorises<ShogiBuckets> for Shogi2344 {
    fn derive_feature(&self, _: &ShogiBuckets, feat: usize) -> Option<usize> {
        Some(feat % 2344)
    }
}

impl Factorises<ShogiBucketsMirrored> for Shogi2344 {
    fn derive_feature(&self, _: &ShogiBucketsMirrored, feat: usize) -> Option<usize> {
        Some(feat % 2344)
    }
}
