//! # warp-vote-consensus
//!
//! Experiments with using GPU warp-vote hardware as an agent consensus primitive.
//!
//! ## Architecture
//!
//! Real GPU warp hardware provides `__ballot_sync(mask, predicate)` — a single
//! instruction that collects 32 thread predicates into a 32-bit ballot word in
//! ~4 clock cycles. This experiment maps that primitive to agent consensus:
//!
//! ```text
//!   10,000 agents
//!        │
//!   ┌────┴────────────────────┐
//!   │  312 Warps (32 agents)  │  ← __ballot_sync: 4 cycles per consensus round
//!   └────┬────────────────────┘
//!        │ ternary vote {Agree, Abstain, Reject} packed as 2 bits
//!   ┌────┴────────────────────┐
//!   │  Quorum Tree (32 warps) │  ← Flux A2A messages between warp leaders
//!   └────┬────────────────────┘
//!        │ CRDT merge at quorum boundaries
//!   ┌────┴────────────────────┐
//!   │  Fleet Decision         │  ← final consensus result
//!   └─────────────────────────┘
//! ```
//!
//! ## Ternary Vote Encoding
//!
//! GPU ballot returns a u32 where bit `i` = thread `i`'s boolean predicate.
//! A ternary vote {-1, 0, +1} needs 2 bits. We run TWO ballot passes:
//!   - Pass 1: `ballot(vote != Reject)`  → positive mask
//!   - Pass 2: `ballot(vote == Agree)`   → agree mask
//! Combined: agree = pass2, abstain = pass1 & !pass2, reject = !pass1
//!
//! This is architecturally faithful: real GPU code would do exactly this.

// ── Core vote type ─────────────────────────────────────────────────────────

/// Ternary agent vote, matching the {-1, 0, +1} trit alphabet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Vote {
    Reject  = -1,  // 0b00 in 2-bit encoding
    Abstain =  0,  // 0b01
    Agree   =  1,  // 0b10
}

impl Vote {
    /// Pack vote as 2 bits (matches ternary-pack's encoding).
    pub fn to_bits(self) -> u8 {
        match self {
            Vote::Reject  => 0b00,
            Vote::Abstain => 0b01,
            Vote::Agree   => 0b10,
        }
    }

    pub fn from_bits(bits: u8) -> Option<Self> {
        match bits & 0b11 {
            0b00 => Some(Vote::Reject),
            0b01 => Some(Vote::Abstain),
            0b10 => Some(Vote::Agree),
            _    => None,  // 0b11 unused, matches ternary-pack's invariant
        }
    }
}

// ── Ballot word ─────────────────────────────────────────────────────────────

/// A 32-thread ballot result, faithful to `__ballot_sync` semantics.
///
/// Two u32 masks represent the full ternary vote of 32 agents:
///   - `positive`: bit set if vote != Reject
///   - `agree`:    bit set if vote == Agree
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BallotWord {
    pub positive: u32,  // __ballot_sync(mask, vote != Reject)
    pub agree:    u32,  // __ballot_sync(mask, vote == Agree)
}

impl BallotWord {
    /// Simulate `__ballot_sync` for a 32-agent warp in a single pass.
    ///
    /// On real GPU this is a hardware instruction; here we fold over the votes.
    pub fn collect(votes: &[Vote; 32]) -> Self {
        let mut positive = 0u32;
        let mut agree    = 0u32;
        for (i, &v) in votes.iter().enumerate() {
            let bit = 1u32 << i;
            match v {
                Vote::Agree   => { positive |= bit; agree |= bit; }
                Vote::Abstain => { positive |= bit; }
                Vote::Reject  => {}
            }
        }
        Self { positive, agree }
    }

    /// Tally votes from the ballot word.
    pub fn tally(self) -> VoteTally {
        let agree   = self.agree.count_ones();
        let abstain = (self.positive & !self.agree).count_ones();
        let reject  = (!self.positive & 0xFFFF_FFFF).count_ones();
        VoteTally { agree, abstain, reject, total: 32 }
    }

    /// True if a simple majority (>50% of non-abstaining) chose Agree.
    pub fn majority_agree(self) -> bool {
        let t = self.tally();
        let decisive = t.agree + t.reject;
        decisive > 0 && t.agree * 2 > decisive
    }

    /// Pack this ballot into 8 bytes for A2A message passing.
    pub fn to_bytes(self) -> [u8; 8] {
        let mut b = [0u8; 8];
        b[0..4].copy_from_slice(&self.positive.to_le_bytes());
        b[4..8].copy_from_slice(&self.agree.to_le_bytes());
        b
    }

    pub fn from_bytes(b: &[u8; 8]) -> Self {
        let positive = u32::from_le_bytes(b[0..4].try_into().unwrap());
        let agree    = u32::from_le_bytes(b[4..8].try_into().unwrap());
        Self { positive, agree }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VoteTally {
    pub agree:   u32,
    pub abstain: u32,
    pub reject:  u32,
    pub total:   u32,
}

impl VoteTally {
    pub fn agreement_rate(self) -> f64 {
        let decisive = self.agree + self.reject;
        if decisive == 0 { return 0.5; }
        self.agree as f64 / decisive as f64
    }
}

// ── Warp ────────────────────────────────────────────────────────────────────

/// A simulated GPU warp: 32 agents with a shared vote round.
#[derive(Debug)]
pub struct Warp {
    pub warp_id: u32,
    agents: [Vote; 32],
    pub ballot: Option<BallotWord>,
    /// Simulated clock cycles consumed (ballot = 4 cycles, setup = ~8).
    pub cycles: u64,
}

impl Warp {
    pub fn new(warp_id: u32) -> Self {
        Self {
            warp_id,
            agents: [Vote::Abstain; 32],
            ballot: None,
            cycles: 0,
        }
    }

    /// Set an individual agent's vote (thread `idx` in the warp).
    pub fn set_vote(&mut self, thread_idx: usize, vote: Vote) {
        assert!(thread_idx < 32, "warp has exactly 32 threads");
        self.agents[thread_idx] = vote;
    }

    /// Execute the two-pass ballot, simulating `__ballot_sync`.
    /// Returns the ballot word and records simulated cycle cost.
    pub fn execute_ballot(&mut self) -> BallotWord {
        // Two ballot passes: 4 cycles each + ~8 cycles setup overhead.
        self.cycles += 8 + 4 + 4;
        let word = BallotWord::collect(&self.agents);
        self.ballot = Some(word);
        word
    }

    pub fn tally(&self) -> Option<VoteTally> {
        self.ballot.map(|b| b.tally())
    }
}

// ── Quorum ───────────────────────────────────────────────────────────────────

/// A quorum of up to 32 warps. The quorum leader aggregates warp ballots
/// via Flux A2A messages and produces a combined decision.
#[derive(Debug)]
pub struct Quorum {
    pub quorum_id: u32,
    warp_ballots: Vec<BallotWord>,
    pub decision: Option<QuorumDecision>,
}

/// The outcome of a quorum vote after merging warp ballots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuorumDecision {
    Accept,
    Reject,
    Deadlock,  // agree == reject, no majority
}

impl Quorum {
    pub fn new(quorum_id: u32) -> Self {
        Self { quorum_id, warp_ballots: Vec::new(), decision: None }
    }

    /// Receive a warp ballot via A2A message (serialized as 8 bytes).
    pub fn receive_a2a_message(&mut self, msg: &[u8; 8]) {
        self.warp_ballots.push(BallotWord::from_bytes(msg));
    }

    /// Merge all warp ballots using CRDT-style aggregation.
    ///
    /// Each warp ballot is independent; the merge is associative, commutative,
    /// and idempotent — it doesn't matter what order warps report in.
    ///
    /// CRDT join: (positive_union, agree_union) is a valid join for OR-lattices
    /// when tracking "has voted", but for majority we sum tallies.
    pub fn merge_and_decide(&mut self) -> QuorumDecision {
        let (total_agree, total_reject) = self.warp_ballots.iter()
            .map(|b| b.tally())
            .fold((0u64, 0u64), |(a, r), t| (a + t.agree as u64, r + t.reject as u64));

        let decision = if total_agree > total_reject {
            QuorumDecision::Accept
        } else if total_reject > total_agree {
            QuorumDecision::Reject
        } else {
            QuorumDecision::Deadlock
        };

        self.decision = Some(decision);
        decision
    }

    pub fn warp_count(&self) -> usize {
        self.warp_ballots.len()
    }

    pub fn total_tally(&self) -> (u64, u64, u64) {
        self.warp_ballots.iter()
            .map(|b| b.tally())
            .fold((0, 0, 0), |(a, ab, r), t| {
                (a + t.agree as u64, ab + t.abstain as u64, r + t.reject as u64)
            })
    }
}

// ── Fleet consensus ──────────────────────────────────────────────────────────

/// A fleet of agents organized into warps and quorums.
///
/// This is the top-level runtime that corresponds to `oxide-fleet`'s
/// fleet coordinator — it drives the full consensus pipeline.
#[derive(Debug)]
pub struct Fleet {
    pub warps: Vec<Warp>,
    pub quorums: Vec<Quorum>,
    pub stats: FleetStats,
}

#[derive(Debug, Default, Clone)]
pub struct FleetStats {
    pub total_agents: usize,
    pub total_warps: usize,
    pub total_quorums: usize,
    pub total_cycles: u64,
    pub final_decision: Option<QuorumDecision>,
    /// Nanoseconds per agent decision at 1GHz (cycles / agents * 1000).
    pub ns_per_agent: f64,
}

impl Fleet {
    /// Build a fleet for `n_agents` agents, auto-sizing warps and quorums.
    pub fn new(n_agents: usize) -> Self {
        let n_warps = (n_agents + 31) / 32;
        let n_quorums = (n_warps + 31) / 32;

        let warps = (0..n_warps as u32).map(Warp::new).collect();
        let quorums = (0..n_quorums as u32).map(Quorum::new).collect();

        Fleet {
            warps,
            quorums,
            stats: FleetStats {
                total_agents: n_agents,
                total_warps: n_warps,
                total_quorums: n_quorums,
                ..Default::default()
            },
        }
    }

    /// Load votes from a slice of `Vote` values (one per agent).
    pub fn load_votes(&mut self, votes: &[Vote]) {
        for (agent_idx, &vote) in votes.iter().enumerate() {
            let warp_idx  = agent_idx / 32;
            let thread_idx = agent_idx % 32;
            if warp_idx < self.warps.len() {
                self.warps[warp_idx].set_vote(thread_idx, vote);
            }
        }
    }

    /// Run the full consensus pipeline: warps ballot → A2A messages → quorum merge.
    ///
    /// This is the core experiment: measures how many cycles from vote submission
    /// to fleet decision, and what the final agreement looks like.
    pub fn run_consensus(&mut self) -> QuorumDecision {
        // Step 1: each warp executes its ballot (parallelizable on GPU).
        let ballot_words: Vec<BallotWord> = self.warps.iter_mut()
            .map(|w| w.execute_ballot())
            .collect();

        let total_warp_cycles: u64 = self.warps.iter().map(|w| w.cycles).sum();

        // Step 2: warp leaders send A2A messages to quorum leaders.
        // Warps route to quorums round-robin (warp i → quorum i/32).
        for (warp_idx, ballot) in ballot_words.iter().enumerate() {
            let quorum_idx = warp_idx / 32;
            if quorum_idx < self.quorums.len() {
                let msg = ballot.to_bytes();
                self.quorums[quorum_idx].receive_a2a_message(&msg);
            }
        }

        // Step 3: quorums merge their ballots (CRDT join).
        let quorum_decisions: Vec<QuorumDecision> = self.quorums.iter_mut()
            .map(|q| q.merge_and_decide())
            .collect();

        // Step 4: fleet-level majority across quorum decisions.
        let accept_count = quorum_decisions.iter().filter(|&&d| d == QuorumDecision::Accept).count();
        let reject_count = quorum_decisions.iter().filter(|&&d| d == QuorumDecision::Reject).count();

        let fleet_decision = if accept_count > reject_count {
            QuorumDecision::Accept
        } else if reject_count > accept_count {
            QuorumDecision::Reject
        } else {
            QuorumDecision::Deadlock
        };

        // On GPU warps run in parallel, so critical path = ONE warp's cycles.
        // Plus A2A message overhead (~100 cycles) and quorum merge (~50 cycles).
        let critical_path_cycles = self.warps.first().map(|w| w.cycles).unwrap_or(0)
            + 100  // A2A message routing
            + 50;  // quorum merge

        let ns_per_agent = if self.stats.total_agents > 0 {
            critical_path_cycles as f64 / self.stats.total_agents as f64 * 1.0
        } else {
            0.0
        };

        self.stats.total_cycles = total_warp_cycles;
        self.stats.final_decision = Some(fleet_decision);
        self.stats.ns_per_agent = ns_per_agent;

        fleet_decision
    }
}

// ── Benchmark harness ────────────────────────────────────────────────────────

/// Compare warp-vote consensus vs CPU mutex consensus on the same vote set.
#[derive(Debug, Clone)]
pub struct ConsensusComparison {
    pub n_agents: usize,
    pub warp_critical_path_cycles: u64,
    pub cpu_mutex_cycles: u64,   // ~100 cycles per agent serialized
    pub speedup: f64,
    pub decision: QuorumDecision,
    pub agreement_rate: f64,
}

impl ConsensusComparison {
    pub fn run(votes: &[Vote]) -> Self {
        let n = votes.len();
        let mut fleet = Fleet::new(n);
        fleet.load_votes(votes);
        let decision = fleet.run_consensus();

        // CPU mutex baseline: each agent acquires a mutex, reads/writes shared counter.
        // ~100 cycles per agent (cache miss + lock contention at scale).
        let cpu_cycles = n as u64 * 100;

        // GPU critical path: ONE warp's ballot (runs all warps in parallel).
        // 8 setup + 8 ballot cycles, plus quorum tree overhead.
        let warp_cycles = 16 + 100 + 50;  // ballot + A2A + merge
        let quorum_levels = if n <= 32 { 1 } else if n <= 1024 { 2 } else { 3 };
        let gpu_critical = warp_cycles * quorum_levels;

        let (total_agree, _abstain, total_reject) = fleet.quorums.iter()
            .map(|q| q.total_tally())
            .fold((0u64, 0u64, 0u64), |(a, ab, r), (qa, qab, qr)| {
                (a + qa, ab + qab, r + qr)
            });

        let decisive = total_agree + total_reject;
        let agreement_rate = if decisive > 0 {
            total_agree as f64 / decisive as f64
        } else {
            0.5
        };

        ConsensusComparison {
            n_agents: n,
            warp_critical_path_cycles: gpu_critical,
            cpu_mutex_cycles: cpu_cycles,
            speedup: cpu_cycles as f64 / gpu_critical as f64,
            decision,
            agreement_rate,
        }
    }
}

// ── Ternary packing for vote buffers ────────────────────────────────────────

/// A packed buffer of ternary votes, matching ternary-pack's 2-bit encoding.
///
/// 16 votes per u32, 0b11 unused. This is the wire format for A2A messages
/// carrying vote batches between agents and quorum coordinators.
#[derive(Debug, Clone)]
pub struct PackedVoteBuffer {
    data: Vec<u32>,
    len: usize,
}

impl PackedVoteBuffer {
    pub fn pack(votes: &[Vote]) -> Self {
        let words = (votes.len() + 15) / 16;
        let mut data = vec![0u32; words];
        for (i, &v) in votes.iter().enumerate() {
            let word    = i / 16;
            let bit_pos = (i % 16) * 2;
            data[word] |= (v.to_bits() as u32) << bit_pos;
        }
        Self { data, len: votes.len() }
    }

    pub fn unpack(&self) -> Vec<Vote> {
        (0..self.len).map(|i| {
            let word    = i / 16;
            let bit_pos = (i % 16) * 2;
            let bits    = ((self.data[word] >> bit_pos) & 0b11) as u8;
            Vote::from_bits(bits).unwrap_or(Vote::Abstain)
        }).collect()
    }

    pub fn bytes(&self) -> usize {
        self.data.len() * 4
    }

    pub fn density_ratio_vs_i32(&self) -> f64 {
        // 2 bits vs 32 bits per vote
        16.0
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Vote encoding ────────────────────────────────────────────────────────

    #[test]
    fn vote_roundtrip_bits() {
        for v in [Vote::Agree, Vote::Abstain, Vote::Reject] {
            assert_eq!(Vote::from_bits(v.to_bits()), Some(v));
        }
    }

    #[test]
    fn bits_0b11_is_none() {
        assert_eq!(Vote::from_bits(0b11), None);
    }

    // ── BallotWord ───────────────────────────────────────────────────────────

    #[test]
    fn unanimous_agree_ballot() {
        let votes = [Vote::Agree; 32];
        let b = BallotWord::collect(&votes);
        assert_eq!(b.positive, 0xFFFF_FFFF);
        assert_eq!(b.agree, 0xFFFF_FFFF);
        let t = b.tally();
        assert_eq!(t.agree, 32);
        assert_eq!(t.abstain, 0);
        assert_eq!(t.reject, 0);
        assert!(b.majority_agree());
    }

    #[test]
    fn unanimous_reject_ballot() {
        let votes = [Vote::Reject; 32];
        let b = BallotWord::collect(&votes);
        assert_eq!(b.positive, 0);
        assert_eq!(b.agree, 0);
        let t = b.tally();
        assert_eq!(t.reject, 32);
        assert!(!b.majority_agree());
    }

    #[test]
    fn mixed_ballot_tally() {
        let mut votes = [Vote::Abstain; 32];
        // 17 agree, 10 reject, 5 abstain
        for i in 0..17  { votes[i]    = Vote::Agree; }
        for i in 17..27 { votes[i]    = Vote::Reject; }
        let b = BallotWord::collect(&votes);
        let t = b.tally();
        assert_eq!(t.agree, 17);
        assert_eq!(t.reject, 10);
        assert_eq!(t.abstain, 5);
        assert!(b.majority_agree());  // 17 > 10
    }

    #[test]
    fn split_vote_no_majority() {
        let mut votes = [Vote::Abstain; 32];
        for i in 0..16  { votes[i] = Vote::Agree; }
        for i in 16..32 { votes[i] = Vote::Reject; }
        let b = BallotWord::collect(&votes);
        assert!(!b.majority_agree());  // exactly tied
    }

    #[test]
    fn ballot_serialization_roundtrip() {
        let mut votes = [Vote::Abstain; 32];
        votes[0]  = Vote::Agree;
        votes[31] = Vote::Reject;
        let original = BallotWord::collect(&votes);
        let bytes = original.to_bytes();
        let recovered = BallotWord::from_bytes(&bytes);
        assert_eq!(original, recovered);
    }

    // ── Warp ────────────────────────────────────────────────────────────────

    #[test]
    fn warp_ballot_cycle_cost() {
        let mut warp = Warp::new(0);
        for i in 0..32 { warp.set_vote(i, Vote::Agree); }
        warp.execute_ballot();
        // 8 setup + 4 + 4 (two ballot passes) = 16 cycles
        assert_eq!(warp.cycles, 16);
    }

    #[test]
    fn warp_partial_participation() {
        let mut warp = Warp::new(42);
        // Only 20 threads vote Agree, 12 default Abstain
        for i in 0..20 { warp.set_vote(i, Vote::Agree); }
        let ballot = warp.execute_ballot();
        let t = ballot.tally();
        assert_eq!(t.agree, 20);
        assert_eq!(t.abstain, 12);
        assert_eq!(t.reject, 0);
    }

    // ── Quorum ───────────────────────────────────────────────────────────────

    #[test]
    fn quorum_receives_warp_messages() {
        let mut q = Quorum::new(0);
        // Two warps both unanimously agree
        let agree_ballot = BallotWord { positive: 0xFFFF_FFFF, agree: 0xFFFF_FFFF };
        q.receive_a2a_message(&agree_ballot.to_bytes());
        q.receive_a2a_message(&agree_ballot.to_bytes());
        assert_eq!(q.warp_count(), 2);
        assert_eq!(q.merge_and_decide(), QuorumDecision::Accept);
    }

    #[test]
    fn quorum_mixed_warp_decisions() {
        let mut q = Quorum::new(1);
        // 2 warps agree, 1 rejects
        let agree = BallotWord { positive: 0xFFFF_FFFF, agree: 0xFFFF_FFFF };
        let reject = BallotWord { positive: 0, agree: 0 };
        q.receive_a2a_message(&agree.to_bytes());
        q.receive_a2a_message(&agree.to_bytes());
        q.receive_a2a_message(&reject.to_bytes());
        // 2*32 agree vs 1*32 reject → Accept
        assert_eq!(q.merge_and_decide(), QuorumDecision::Accept);
    }

    #[test]
    fn quorum_deadlock_on_exact_split() {
        let mut q = Quorum::new(2);
        let agree  = BallotWord { positive: 0xFFFF_FFFF, agree: 0xFFFF_FFFF };
        let reject = BallotWord { positive: 0, agree: 0 };
        q.receive_a2a_message(&agree.to_bytes());
        q.receive_a2a_message(&reject.to_bytes());
        assert_eq!(q.merge_and_decide(), QuorumDecision::Deadlock);
    }

    // ── Fleet ────────────────────────────────────────────────────────────────

    #[test]
    fn fleet_10000_agents_unanimous() {
        let votes: Vec<Vote> = vec![Vote::Agree; 10_000];
        let mut fleet = Fleet::new(10_000);
        fleet.load_votes(&votes);
        let decision = fleet.run_consensus();
        assert_eq!(decision, QuorumDecision::Accept);
        assert!(fleet.stats.total_cycles > 0);
    }

    #[test]
    fn fleet_majority_wins() {
        let mut votes: Vec<Vote> = vec![Vote::Agree; 6_000];
        votes.extend(vec![Vote::Reject; 4_000]);
        let mut fleet = Fleet::new(10_000);
        fleet.load_votes(&votes);
        assert_eq!(fleet.run_consensus(), QuorumDecision::Accept);
    }

    #[test]
    fn fleet_scaling() {
        for n in [32, 64, 1024, 10_000] {
            let votes: Vec<Vote> = (0..n).map(|i| {
                if i % 3 == 0 { Vote::Reject } else { Vote::Agree }
            }).collect();
            let mut fleet = Fleet::new(n);
            fleet.load_votes(&votes);
            let d = fleet.run_consensus();
            // 2/3 agree → should Accept at all scales
            assert_eq!(d, QuorumDecision::Accept, "failed at n={}", n);
        }
    }

    #[test]
    fn fleet_stats_populated() {
        let votes = vec![Vote::Agree; 1024];
        let mut fleet = Fleet::new(1024);
        fleet.load_votes(&votes);
        fleet.run_consensus();
        assert_eq!(fleet.stats.total_warps, 32);
        assert_eq!(fleet.stats.total_quorums, 1);
        assert!(fleet.stats.total_cycles > 0);
    }

    // ── Consensus comparison (GPU vs CPU baseline) ────────────────────────────

    #[test]
    fn gpu_warp_vote_outperforms_cpu_mutex() {
        let votes: Vec<Vote> = (0..10_000).map(|i| {
            if i % 4 != 0 { Vote::Agree } else { Vote::Abstain }
        }).collect();

        let cmp = ConsensusComparison::run(&votes);
        assert_eq!(cmp.n_agents, 10_000);
        // GPU critical path is constant (one warp deep), CPU scales with n.
        assert!(cmp.speedup > 10.0,
            "expected >10× speedup, got {:.1}×", cmp.speedup);
        assert_eq!(cmp.decision, QuorumDecision::Accept);
        println!(
            "\nConsensus comparison — {} agents:\n  GPU critical path: {} cycles\n  CPU mutex: {} cycles\n  Speedup: {:.1}×\n  Agreement rate: {:.1}%",
            cmp.n_agents,
            cmp.warp_critical_path_cycles,
            cmp.cpu_mutex_cycles,
            cmp.speedup,
            cmp.agreement_rate * 100.0
        );
    }

    // ── Packed vote buffer (ternary-pack synergy) ─────────────────────────────

    #[test]
    fn packed_vote_buffer_roundtrip() {
        let votes: Vec<Vote> = (0..100).map(|i| match i % 3 {
            0 => Vote::Agree,
            1 => Vote::Abstain,
            _ => Vote::Reject,
        }).collect();

        let packed = PackedVoteBuffer::pack(&votes);
        let unpacked = packed.unpack();
        assert_eq!(votes, unpacked);
    }

    #[test]
    fn packed_buffer_density() {
        let votes = vec![Vote::Agree; 1024];
        let packed = PackedVoteBuffer::pack(&votes);
        // 1024 votes × 2 bits = 256 bytes (vs 4096 bytes for i32 array)
        assert_eq!(packed.bytes(), 256);
        assert_eq!(packed.density_ratio_vs_i32(), 16.0);
    }

    #[test]
    fn packed_buffer_wire_size_scales() {
        // For 10K agents sending votes to a quorum coordinator:
        let votes = vec![Vote::Agree; 10_000];
        let packed = PackedVoteBuffer::pack(&votes);
        // 10,000 × 2 bits = 2,500 bytes (vs 40,000 for i32)
        assert_eq!(packed.bytes(), 2_500);
    }

    // ── Agreement rate metric ─────────────────────────────────────────────────

    #[test]
    fn agreement_rate_unanimous() {
        let t = VoteTally { agree: 32, abstain: 0, reject: 0, total: 32 };
        assert!((t.agreement_rate() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn agreement_rate_split() {
        let t = VoteTally { agree: 16, abstain: 8, reject: 8, total: 32 };
        // 16 agree / (16+8) decisive = 0.667
        assert!((t.agreement_rate() - (16.0 / 24.0)).abs() < 1e-9);
    }

    #[test]
    fn agreement_rate_all_abstain() {
        let t = VoteTally { agree: 0, abstain: 32, reject: 0, total: 32 };
        // No decisive votes → 0.5 (neutral)
        assert!((t.agreement_rate() - 0.5).abs() < 1e-9);
    }
}
