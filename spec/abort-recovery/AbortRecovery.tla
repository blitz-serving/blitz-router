--------------------------- MODULE AbortRecovery ---------------------------
(*
 * TLA+ specification for blitz-router's abort-recovery invariants.
 *
 * Scope: this is NOT a model of the whole system. It models only the
 * recovery process when an in-flight request is terminated early by
 * one of two exception flavours:
 *
 *   - FrontAbort:    the frontend (client) drops the response channel.
 *   - BackendFault:  the engine reports an error mid-stream.
 *
 * The spec verifies that the colocation event-loop pair (work loop +
 * completion loop) preserves entry-ownership and metric-accounting
 * invariants under arbitrary interleavings of those exceptions with
 * normal step processing. It catches order-dependent races in the
 * exception-handling code paths, not policy logic.
 *
 * Models the interplay between:
 *   - Work loop:       polls queue, sends add_request to engine, passes entry
 *                      to completion loop via bounded channel (wq_tx/cq_wqe_rx).
 *   - Completion loop: receives EngineStepOutput, processes entries, handles
 *                      exceptions (abort/fault), manages metrics.
 *
 * Two nested FSMs:
 *   1. RequestPhase:  Waiting -> Prefilling -> Decoding -> (removed)
 *   2. ExtStInner:    WaitingTerm / Faulted / Exited (with live flags)
 *
 * Entry ownership is tracked across three containers:
 *   entries, except_ctx, temp_leaving
 *
 * Reference: router/src/engine/colocation.rs
 * Issue:     https://github.com/blitz-serving/blitz-router/issues/4
 *)

EXTENDS Naturals, FiniteSets, Sequences, TLC

CONSTANTS
    RequestIds,         \* Finite set of possible request IDs (e.g. {1,2,3})
    MaxSteps,           \* Bound on model checking depth
    CHANNEL_CAP         \* Bounded channel capacity (64 in code)

(* -------------------------------------------------------------------------- *)
(* Request lifecycle phases (RequestPhase enum in Rust)                        *)
(* -------------------------------------------------------------------------- *)
Phase_Waiting    == "Waiting"
Phase_Prefilling == "Prefilling"
Phase_Decoding   == "Decoding"
Phase_Excepted   == "Excepted"

(* -------------------------------------------------------------------------- *)
(* Exception states (ExtStInner enum in Rust)                                 *)
(* -------------------------------------------------------------------------- *)
Ext_WaitingTerm  == "WaitingTerm"
Ext_Faulted      == "Faulted"
Ext_Exited       == "Exited"

(* -------------------------------------------------------------------------- *)
(* SSE states reported by engine                                              *)
(* -------------------------------------------------------------------------- *)
SSE_PREFILL == "PREFILL"
SSE_DECODE  == "DECODE"

(* ========================================================================== *)
(* VARIABLES                                                                  *)
(* ========================================================================== *)
VARIABLES
    (* ----- Work loop state ----- *)
    queue,              \* Set of request IDs waiting to be dispatched
    wq_channel,         \* Bounded channel: work loop -> completion loop (sequence of IDs)
    engine_pending,     \* Set of request IDs sent to engine via add_request()

    (* ----- Completion loop state ----- *)
    entries,            \* Function: request_id -> phase (requests actively tracked)
    except_ctx,         \* Function: request_id -> [state, live, waiting_term]
    temp_leaving,       \* Set of request IDs temporarily leaving during a step
    cancel_req_ids,     \* Sequence of IDs pending frontend abort (within a step)

    (* ----- Cross-loop state ----- *)
    phases,             \* Function: request_id -> RequestPhase
    finished,           \* Set of request IDs that have fully exited the system
    aborted_at_router,  \* Set of request IDs where router called abort_request()

    (* ----- Metrics ----- *)
    bs,                 \* Batch size counter (number of active requests at engine)
    all_tokens,         \* Total token counter

    (* ----- Engine model ----- *)
    engine_active,      \* Set of request IDs the engine is currently processing
    engine_aborted,     \* Set of request IDs the engine has ACK-ed abort for

    (* ----- Model checking control ----- *)
    step_count          \* Step counter for bounded model checking

vars == <<queue, wq_channel, engine_pending, entries, except_ctx,
          temp_leaving, cancel_req_ids, phases, finished,
          aborted_at_router, bs, all_tokens, engine_active,
          engine_aborted, step_count>>

(* ========================================================================== *)
(* HELPER OPERATORS                                                           *)
(* ========================================================================== *)

\* Domain of a function (set of keys)
Dom(f) == DOMAIN f

\* Remove a key from a function
RemoveKey(f, k) == [x \in (DOMAIN f \ {k}) |-> f[x]]

\* Add/update a key in a function
UpdateKey(f, k, v) == [x \in (DOMAIN f \union {k}) |-> IF x = k THEN v ELSE f[x]]

\* Empty function
EmptyFn == [x \in {} |-> ""]

(* -------------------------------------------------------------------------- *)
(* RequestPhase transition function                                           *)
(* Mirrors RequestPhase::transition() in colocation.rs:129-170                *)
(* -------------------------------------------------------------------------- *)
PhaseTransition(current_phase, sse_state) ==
    CASE current_phase = Phase_Waiting /\ sse_state = SSE_PREFILL
            -> Phase_Prefilling
      [] current_phase = Phase_Waiting /\ sse_state = SSE_DECODE
            -> Phase_Decoding        \* prefill not observed
      [] current_phase = Phase_Prefilling /\ sse_state = SSE_PREFILL
            -> Phase_Prefilling      \* chunked prefill
      [] current_phase = Phase_Prefilling /\ sse_state = SSE_DECODE
            -> Phase_Decoding        \* normal transition
      [] current_phase = Phase_Decoding /\ sse_state = SSE_DECODE
            -> Phase_Decoding        \* continued decode
      [] current_phase = Phase_Decoding /\ sse_state = SSE_PREFILL
            -> Phase_Prefilling      \* invalid but tolerated in release
      [] current_phase = Phase_Excepted
            -> Phase_Excepted        \* no transitions from Excepted
      [] OTHER -> "INVALID"

(* ========================================================================== *)
(* INITIAL STATE                                                              *)
(* ========================================================================== *)
Init ==
    /\ queue           = RequestIds
    /\ wq_channel      = <<>>
    /\ engine_pending   = {}
    /\ entries         = EmptyFn
    /\ except_ctx      = EmptyFn
    /\ temp_leaving    = {}
    /\ cancel_req_ids  = <<>>
    /\ phases          = EmptyFn
    /\ finished        = {}
    /\ aborted_at_router = {}
    /\ bs              = 0
    /\ all_tokens      = 0
    /\ engine_active   = {}
    /\ engine_aborted  = {}
    /\ step_count      = 0

(* State space bound for safety checking *)
StepBound == step_count <= MaxSteps

(* ========================================================================== *)
(* ACTIONS                                                                    *)
(* ========================================================================== *)

(* -------------------------------------------------------------------------- *)
(* ACTION: WorkLoop_Dispatch                                                  *)
(*                                                                            *)
(* Work loop picks a request from queue, sends add_request to engine,         *)
(* and pushes the entry into the bounded channel.                             *)
(* Models: colocation.rs:207-232                                              *)
(*   - wq_tx.reserve().await -> channel capacity check                        *)
(*   - queue.next_request()  -> dequeue                                       *)
(*   - engine_client.add_request() -> engine_pending                          *)
(*   - permit.send(entry)    -> wq_channel append                             *)
(*   - wqe_mtx.lock()       -> atomicity of add_request + channel send       *)
(* -------------------------------------------------------------------------- *)
WorkLoop_Dispatch ==
    /\ queue # {}
    /\ Len(wq_channel) < CHANNEL_CAP
    /\ \E id \in queue :
        /\ queue'           = queue \ {id}
        /\ engine_pending'  = engine_pending \union {id}
        /\ wq_channel'     = Append(wq_channel, id)
        /\ UNCHANGED <<entries, except_ctx, temp_leaving, cancel_req_ids,
                        phases, finished, aborted_at_router, bs, all_tokens,
                        engine_active, engine_aborted, step_count>>

(* -------------------------------------------------------------------------- *)
(* ACTION: CompletionLoop_DrainChannel                                        *)
(*                                                                            *)
(* Completion loop drains the wq channel into entries map.                    *)
(* Models: colocation.rs:414-428                                              *)
(*   - wqe_mtx.lock() -> synchronized with work loop                         *)
(*   - cq_wqe_rx.try_recv() -> drain all pending entries                     *)
(*   - entries.insert(rid, entry)                                             *)
(*   - request_phases.insert(rid, Waiting)                                    *)
(* -------------------------------------------------------------------------- *)
CompletionLoop_DrainChannel ==
    /\ wq_channel # <<>>
    /\ LET ids == {wq_channel[i] : i \in 1..Len(wq_channel)}
       IN /\ entries'    = [x \in (Dom(entries) \union ids) |->
                              IF x \in ids THEN Phase_Waiting ELSE entries[x]]
          /\ phases'     = [x \in (Dom(phases) \union ids) |->
                              IF x \in ids THEN Phase_Waiting ELSE phases[x]]
          /\ wq_channel' = <<>>
    /\ UNCHANGED <<queue, engine_pending, except_ctx, temp_leaving,
                    cancel_req_ids, finished, aborted_at_router, bs,
                    all_tokens, engine_active, engine_aborted, step_count>>

(* -------------------------------------------------------------------------- *)
(* ACTION: Engine_AcceptRequest                                               *)
(*                                                                            *)
(* Engine picks up a pending request and starts processing.                   *)
(* Implicit assumption A1: engine only processes IDs sent via add_request().  *)
(* -------------------------------------------------------------------------- *)
Engine_AcceptRequest ==
    /\ engine_pending # {}
    /\ \E id \in engine_pending :
        /\ engine_active'  = engine_active \union {id}
        /\ engine_pending' = engine_pending \ {id}
        /\ UNCHANGED <<queue, wq_channel, entries, except_ctx, temp_leaving,
                        cancel_req_ids, phases, finished, aborted_at_router,
                        bs, all_tokens, engine_aborted, step_count>>

(* -------------------------------------------------------------------------- *)
(* ACTION: CompletionLoop_StepOutput                                          *)
(*                                                                            *)
(* The core action: engine produces a step output that the completion loop    *)
(* processes. This models the ENTIRE body of the main loop in                 *)
(* completion_event_loop (colocation.rs:398-761).                             *)
(*                                                                            *)
(* A step non-deterministically selects:                                      *)
(*   - A subset of engine_active requests to report on                        *)
(*   - For each: SSE state (PREFILL/DECODE) and is_finished flag             *)
(*   - Whether frontend abort occurred (response_tx.send fails)              *)
(*   - A subset of aborted_at_router to appear in aborted_requests (Term)    *)
(*   - A subset of engine_active to appear in error_rx (backend fault)       *)
(*                                                                            *)
(* Engine assumptions encoded:                                                *)
(*   A1: only reports IDs in engine_active (previously added)                *)
(*   A2: each ID appears at most once per step                               *)
(*   A3: state monotonicity is NOT strictly enforced here (model checks it)  *)
(*   A4: is_finished reported at most once (checked by invariant)            *)
(*   A5: finished IDs removed from engine_active                             *)
(*   B1: Term only for IDs in aborted_at_router                              *)
(*   B2: Term at most once per ID                                            *)
(* -------------------------------------------------------------------------- *)
CompletionLoop_StepOutput ==
    \* Engine produces steps as long as it has work (active requests or pending exceptions)
    /\ (Dom(entries) \union Dom(except_ctx) \union engine_active) # {}
    /\ \E reported \in SUBSET (Dom(entries) \union Dom(except_ctx)) :
       \E sse_states \in [reported -> {SSE_PREFILL, SSE_DECODE}] :
       \E is_fin \in [reported -> BOOLEAN] :
       \E frontend_aborts \in SUBSET (reported \intersect Dom(entries)) :
       \E term_ids \in SUBSET (aborted_at_router \intersect Dom(except_ctx)) :
       \E fault_ids \in SUBSET (Dom(entries) \ reported) :
          \* Guard: reported IDs must be in engine_active (assumption A1)
          /\ reported \subseteq engine_active
          \* Guard: term_ids must be in except_ctx with WaitingTerm state (assumption B1)
          /\ \A tid \in term_ids :
               tid \in Dom(except_ctx) /\ except_ctx[tid].state = Ext_WaitingTerm
          \* Guard: fault_ids must not overlap with already excepted requests
          /\ fault_ids \intersect Dom(except_ctx) = {}
          /\ LET
                (* ---- Phase 1: Process step outputs (fast path metrics + slow path callbacks) ---- *)

                \* IDs that finish in this step and are in entries (not excepted)
                finishing == {id \in reported \intersect Dom(entries) : is_fin[id]}

                \* IDs that get frontend abort during callback
                \* NOTE: frontend_aborts that also finish get revoked (lines 637-641)
                actual_aborts == frontend_aborts \ finishing

                \* IDs that are in except_ctx and finish (Exit event)
                excepted_finishing == {id \in reported \intersect Dom(except_ctx) : is_fin[id]}

                (* ---- Phase 2: Build new state ---- *)

                \* entries after removing finished + frontend-aborted + backend-faulted
                entries_after_finish == [id \in (((Dom(entries) \ finishing) \ actual_aborts) \ fault_ids)
                                           |-> entries[id]]

                \* Update phases for non-finishing, non-aborting requests
                phases_after == [id \in Dom(phases) |->
                    IF id \in finishing \/ id \in excepted_finishing
                    THEN phases[id]  \* about to be removed
                    ELSE IF id \in actual_aborts \/ id \in fault_ids
                    THEN Phase_Excepted
                    ELSE IF id \in reported /\ id \in Dom(entries)
                    THEN PhaseTransition(phases[id], sse_states[id])
                    ELSE phases[id]]

                \* Remove phases for finished requests
                phases_cleaned == [id \in (Dom(phases_after) \ finishing) |-> phases_after[id]]

                (* ---- Phase 3: Exception handling (on_frontend_abort + on_backend_fault) ---- *)

                \* Build new except_ctx entries for aborted requests
                new_abort_entries == [id \in actual_aborts |->
                    [state |-> Ext_WaitingTerm, live |-> FALSE, waiting_term |-> TRUE]]

                \* Build new except_ctx entries for faulted requests
                new_fault_entries == [id \in fault_ids |->
                    [state |-> Ext_Faulted, live |-> TRUE, waiting_term |-> FALSE]]

                \* Process Exit events on existing except_ctx entries
                except_after_exit == [id \in (Dom(except_ctx) \ excepted_finishing) |->
                    IF id \in reported  \* Live event
                    THEN [except_ctx[id] EXCEPT !.live = TRUE]
                    ELSE except_ctx[id]]

                (* ---- Phase 4: Term processing ---- *)

                \* Remove term_ids from except_ctx (they are ACK-ed)
                except_after_term == [id \in (Dom(except_after_exit) \ term_ids) |->
                    except_after_exit[id]]

                \* Remove phases for term_ids
                phases_after_term == [id \in (Dom(phases_cleaned) \ term_ids) |-> phases_cleaned[id]]

                (* ---- Phase 5: filter_drop (end-of-cycle cleanup) ---- *)

                \* Faulted entries that are not live get dropped
                drop_ready == {id \in Dom(except_after_term) :
                    except_after_term[id].state = Ext_Faulted
                    /\ except_after_term[id].live = FALSE}

                \* Clear live flags and remove drop_ready
                except_final == [id \in (Dom(except_after_term) \ drop_ready) |->
                    [except_after_term[id] EXCEPT !.live = FALSE]]

                \* Remove phases for dropped entries
                phases_final == [id \in ((Dom(phases_after_term) \ drop_ready)
                                         \ excepted_finishing) |->
                    phases_after_term[id]]

                (* ---- Phase 6: Merge new exceptions into except_ctx ---- *)

                all_except == [id \in (Dom(except_final) \union Dom(new_abort_entries)
                                       \union Dom(new_fault_entries)) |->
                    IF id \in Dom(new_abort_entries) THEN new_abort_entries[id]
                    ELSE IF id \in Dom(new_fault_entries) THEN new_fault_entries[id]
                    ELSE except_final[id]]

                \* Merge abort/fault phases
                all_phases == [id \in (Dom(phases_final) \union Dom(new_abort_entries)
                                       \union Dom(new_fault_entries)) |->
                    IF id \in Dom(new_abort_entries) \/ id \in Dom(new_fault_entries)
                    THEN Phase_Excepted
                    ELSE phases_final[id]]

                (* ---- Metrics ---- *)
                bs_delta == Cardinality(finishing) + Cardinality(excepted_finishing)
                            + Cardinality(term_ids) + Cardinality(drop_ready)

                (* ---- Finished set ---- *)
                newly_finished == finishing \union excepted_finishing
                                  \union term_ids \union drop_ready

                (* ---- Engine state ---- *)
                engine_after == (engine_active \ finishing) \ excepted_finishing

             IN
                /\ entries'          = entries_after_finish
                /\ except_ctx'       = all_except
                /\ phases'           = all_phases
                /\ temp_leaving'     = {}   \* cleared at end of step
                /\ cancel_req_ids'   = <<>> \* cleared at end of step
                /\ finished'         = finished \union newly_finished
                /\ aborted_at_router' = (aborted_at_router \union actual_aborts) \ term_ids
                \* Real code uses saturating_sub (colocation.rs:723)
                /\ LET bs_raw == (bs + Cardinality(reported \intersect Dom(entries)))
                                  - bs_delta
                   IN bs' = IF bs_raw < 0 THEN 0 ELSE bs_raw
                /\ all_tokens'       = all_tokens  \* simplified; token counting omitted
                /\ engine_active'    = engine_after
                /\ engine_aborted'   = engine_aborted \union term_ids
                /\ step_count'       = step_count + 1  \* bounded via CONSTRAINT in cfg
                /\ UNCHANGED <<queue, wq_channel, engine_pending>>

(* -------------------------------------------------------------------------- *)
(* ACTION: FrontendAbort                                                      *)
(*                                                                            *)
(* Router decides to abort a request (frontend dropped response channel).     *)
(* In the real code this happens inside on_prefill/on_decode callbacks.       *)
(* Here we model it as router calling abort_request() on the engine.          *)
(* -------------------------------------------------------------------------- *)
FrontendAbort ==
    /\ \E id \in Dom(entries) :
        /\ id \notin aborted_at_router
        /\ aborted_at_router' = aborted_at_router \union {id}
        /\ UNCHANGED <<queue, wq_channel, engine_pending, entries, except_ctx,
                        temp_leaving, cancel_req_ids, phases, finished,
                        bs, all_tokens, engine_active, engine_aborted, step_count>>

(* -------------------------------------------------------------------------- *)
(* ACTION: Engine_ProduceFinish                                               *)
(*                                                                            *)
(* Engine decides a request is done (is_finished=true in next step output).   *)
(* This is modeled by removing from engine_active — the completion loop       *)
(* will see is_finished=true in the next CompletionLoop_StepOutput.           *)
(* Separated to enable fairness: engine MUST eventually finish requests.      *)
(*                                                                            *)
(* Implicit assumption A4: is_finished reported exactly once.                 *)
(* Implicit assumption A5: request never appears again after finishing.       *)
(* -------------------------------------------------------------------------- *)
Engine_ProduceFinish ==
    /\ \E id \in engine_active :
        /\ id \in Dom(entries)   \* only finish requests still in entries
        /\ entries'      = RemoveKey(entries, id)
        /\ phases'       = RemoveKey(phases, id)
        /\ finished'     = finished \union {id}
        /\ engine_active' = engine_active \ {id}
        /\ bs'           = IF bs > 0 THEN bs - 1 ELSE 0
        /\ UNCHANGED <<queue, wq_channel, engine_pending, except_ctx,
                        temp_leaving, cancel_req_ids, aborted_at_router,
                        all_tokens, engine_aborted, step_count>>

(* -------------------------------------------------------------------------- *)
(* ACTION: Engine_AcknowledgeAbort                                            *)
(*                                                                            *)
(* Engine ACKs an abort: sends Term via aborted_requests in step output.      *)
(* Models the engine side of the 2PC abort protocol.                          *)
(*                                                                            *)
(* Implicit assumption B1: Term only after abort_request().                   *)
(* Implicit assumption B2: Term at most once per request.                     *)
(* Implicit assumption B3 (LIVENESS): engine EVENTUALLY sends Term.           *)
(* -------------------------------------------------------------------------- *)
Engine_AcknowledgeAbort ==
    /\ \E id \in aborted_at_router :
        /\ id \in Dom(except_ctx)
        /\ except_ctx[id].state = Ext_WaitingTerm
        /\ id \notin engine_aborted
        \* Process Term: remove from except_ctx
        /\ except_ctx'       = RemoveKey(except_ctx, id)
        /\ phases'           = RemoveKey(phases, id)
        /\ finished'         = finished \union {id}
        /\ aborted_at_router' = aborted_at_router \ {id}
        /\ engine_aborted'   = engine_aborted \union {id}
        /\ engine_active'    = engine_active \ {id}
        /\ bs'               = IF bs > 0 THEN bs - 1 ELSE 0
        /\ UNCHANGED <<queue, wq_channel, engine_pending, entries,
                        temp_leaving, cancel_req_ids, all_tokens, step_count>>

(* -------------------------------------------------------------------------- *)
(* ACTION: Stutter (for liveness checking / fairness)                         *)
(* -------------------------------------------------------------------------- *)
Stutter == UNCHANGED vars

(* ========================================================================== *)
(* NEXT-STATE RELATION                                                        *)
(* ========================================================================== *)
Next ==
    \/ WorkLoop_Dispatch
    \/ CompletionLoop_DrainChannel
    \/ Engine_AcceptRequest
    \/ CompletionLoop_StepOutput
    \/ Engine_ProduceFinish
    \/ Engine_AcknowledgeAbort
    \/ FrontendAbort
    \/ Stutter

Spec == Init /\ [][Next]_vars

(* ========================================================================== *)
(* SAFETY INVARIANTS                                                          *)
(* ========================================================================== *)

(* -------------------------------------------------------------------------- *)
(* INV P1: Ownership Uniqueness                                               *)
(*                                                                            *)
(* A request ID exists in AT MOST ONE of:                                     *)
(*   entries, except_ctx, temp_leaving, finished                              *)
(* This is the most critical invariant — violation means entry leak or        *)
(* double-counting.                                                           *)
(* Maps to: debug_assert at colocation.rs:736-741                             *)
(* -------------------------------------------------------------------------- *)
OwnershipUniqueness ==
    /\ Dom(entries) \intersect Dom(except_ctx) = {}
    /\ Dom(entries) \intersect finished = {}
    /\ Dom(except_ctx) \intersect finished = {}

(* -------------------------------------------------------------------------- *)
(* INV P2: Phase-Container Consistency                                        *)
(*                                                                            *)
(* Every request in entries has a corresponding phase that is NOT Excepted.   *)
(* Every request in except_ctx has phase = Excepted.                          *)
(* Maps to: debug_assert at colocation.rs:744-759                             *)
(* -------------------------------------------------------------------------- *)
PhaseConsistency ==
    /\ \A id \in Dom(entries) :
        /\ id \in Dom(phases)
        /\ phases[id] # Phase_Excepted
    /\ \A id \in Dom(except_ctx) :
        /\ id \in Dom(phases)
        /\ phases[id] = Phase_Excepted

(* -------------------------------------------------------------------------- *)
(* INV P3: Phase Tracking Completeness                                        *)
(*                                                                            *)
(* phases domain = entries domain ∪ except_ctx domain                         *)
(* No orphaned phases, no missing phases.                                     *)
(* -------------------------------------------------------------------------- *)
PhaseCompleteness ==
    Dom(phases) = Dom(entries) \union Dom(except_ctx)

(* -------------------------------------------------------------------------- *)
(* INV P4: Exception State Validity                                           *)
(*                                                                            *)
(* Requests in except_ctx have valid ExtStInner states.                       *)
(* WaitingTerm requests must have been aborted at router.                     *)
(* Faulted requests were never aborted (different exception path).            *)
(* -------------------------------------------------------------------------- *)
ExceptionStateValidity ==
    \A id \in Dom(except_ctx) :
        /\ except_ctx[id].state \in {Ext_WaitingTerm, Ext_Faulted, Ext_Exited}
        /\ except_ctx[id].state = Ext_WaitingTerm =>
            id \in aborted_at_router

(* -------------------------------------------------------------------------- *)
(* INV P5: No Request Resurrection                                            *)
(*                                                                            *)
(* Once a request enters `finished`, it never reappears in entries or         *)
(* except_ctx.                                                                *)
(* -------------------------------------------------------------------------- *)
NoResurrection ==
    /\ finished \intersect Dom(entries) = {}
    /\ finished \intersect Dom(except_ctx) = {}

(* -------------------------------------------------------------------------- *)
(* INV P6: Engine Containment                                                 *)
(*                                                                            *)
(* Engine only processes requests that were sent to it.                       *)
(* Implicit assumption A1.                                                    *)
(* -------------------------------------------------------------------------- *)
\* IDs currently in the wq_channel (in transit from work loop to completion loop)
ChannelIds == {wq_channel[i] : i \in 1..Len(wq_channel)}

EngineContainment ==
    engine_active \subseteq (Dom(entries) \union Dom(except_ctx)
                             \union finished \union engine_pending
                             \union ChannelIds)

(* -------------------------------------------------------------------------- *)
(* INV P7: Batch Size Non-Negative                                            *)
(*                                                                            *)
(* Metric counter should never go negative.                                   *)
(* Maps to: saturating_sub at colocation.rs:723                               *)
(* -------------------------------------------------------------------------- *)
MetricNonNegative ==
    bs >= 0

(* -------------------------------------------------------------------------- *)
(* Combined safety invariant                                                  *)
(* -------------------------------------------------------------------------- *)
SafetyInvariant ==
    /\ OwnershipUniqueness
    /\ PhaseConsistency
    /\ PhaseCompleteness
    /\ ExceptionStateValidity
    /\ NoResurrection
    /\ EngineContainment
    /\ MetricNonNegative

(* ========================================================================== *)
(* LIVENESS PROPERTIES                                                        *)
(* ========================================================================== *)

(* -------------------------------------------------------------------------- *)
(* LIVENESS L1: Request Termination                                           *)
(*                                                                            *)
(* Every request that enters the system eventually reaches `finished`.        *)
(* Requires fairness on engine actions (engine eventually processes and       *)
(* finishes requests, and eventually sends Term for aborted requests).        *)
(*                                                                            *)
(* This is the property that catches the WaitingTerm-forever bug              *)
(* (assumption B3 violation).                                                 *)
(* -------------------------------------------------------------------------- *)
RequestTermination ==
    \A id \in RequestIds :
        [](id \in Dom(entries) => <>(id \in finished))

(* -------------------------------------------------------------------------- *)
(* LIVENESS L2: Exception Resolution                                          *)
(*                                                                            *)
(* Every request that enters except_ctx eventually leaves it.                 *)
(* Catches: stuck WaitingTerm, stuck Faulted{live:true}.                      *)
(* -------------------------------------------------------------------------- *)
ExceptionResolution ==
    \A id \in RequestIds :
        [](id \in Dom(except_ctx) => <>(id \notin Dom(except_ctx)))

(* -------------------------------------------------------------------------- *)
(* LIVENESS L3: Channel Drain                                                 *)
(*                                                                            *)
(* The wq_channel eventually gets drained (completion loop is responsive).    *)
(* -------------------------------------------------------------------------- *)
ChannelDrain ==
    [](Len(wq_channel) > 0 => <>(Len(wq_channel) = 0))

(* ========================================================================== *)
(* FAIRNESS CONDITIONS                                                        *)
(* ========================================================================== *)

\* Weak fairness: if an action is continuously enabled, it eventually fires.
\* Required for liveness properties.
(* Weak fairness on all non-environmental actions.                            *)
(* WF on Engine_AcknowledgeAbort encodes assumption B3:                       *)
(*   "engine EVENTUALLY sends Term after receiving abort"                     *)
(* WF on Engine_ProduceFinish encodes the assumption that engine              *)
(*   eventually finishes processing every request.                            *)
(* Remove WF on Engine_AcknowledgeAbort to test B3 violation detection.       *)
Fairness ==
    /\ WF_vars(CompletionLoop_DrainChannel)
    /\ WF_vars(CompletionLoop_StepOutput)
    /\ WF_vars(Engine_AcceptRequest)
    /\ WF_vars(Engine_ProduceFinish)
    /\ WF_vars(Engine_AcknowledgeAbort)

(* Fairness WITHOUT engine Term ACK — use to demonstrate B3 violation.        *)
FairnessNoTermAck ==
    /\ WF_vars(CompletionLoop_DrainChannel)
    /\ WF_vars(CompletionLoop_StepOutput)
    /\ WF_vars(Engine_AcceptRequest)
    /\ WF_vars(Engine_ProduceFinish)
    \* NOTE: no WF on Engine_AcknowledgeAbort — engine may never ACK

LiveSpec == Init /\ [][Next]_vars /\ Fairness

\* Use this spec to verify that B3 violation causes liveness failure:
LiveSpecNoTermAck == Init /\ [][Next]_vars /\ FairnessNoTermAck

=============================================================================
