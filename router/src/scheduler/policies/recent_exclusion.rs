use std::collections::VecDeque;

const RECENT_REPLICA_WINDOW: usize = 3;

#[derive(Debug, Default)]
pub(crate) struct RecentReplicaExclusion {
    recent: VecDeque<usize>,
}

impl RecentReplicaExclusion {
    pub(super) fn is_excluded(&self, replica_idx: usize) -> bool {
        self.recent.contains(&replica_idx)
    }

    pub(super) fn has_allowed_replica(&self, num_replicas: usize) -> bool {
        (0..num_replicas).any(|idx| !self.is_excluded(idx))
    }

    pub(super) fn record(&mut self, replica_idx: usize) {
        self.recent.push_back(replica_idx);
        while self.recent.len() > RECENT_REPLICA_WINDOW {
            self.recent.pop_front();
        }
    }

    pub(super) fn as_vec(&self) -> Vec<usize> {
        self.recent.iter().copied().collect()
    }
}
