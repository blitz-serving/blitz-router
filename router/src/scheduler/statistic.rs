use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use tokio::fs::File;
use tokio::io::{AsyncWriteExt, BufWriter};
use tokio::sync::Mutex;
use tokio::time::{Duration, sleep};

use crate::ScheduleContext;

const STATISIC_INTERVAL: u64 = 2;
static PREFILL_TOKENS: AtomicUsize = AtomicUsize::new(0);

pub(crate) fn increase_prefill_tokens(count: usize) {
    PREFILL_TOKENS.fetch_add(count, Ordering::SeqCst);
}


pub(crate) async fn statistic(
    all_schedule_contexts: Vec<Arc<Mutex<ScheduleContext>>>,
    statistic_path: Option<String>,
) {
    if statistic_path.is_none() {
        tracing::warn!("Statistic path is not set");
        return;
    }
    
    let file = File::create(statistic_path.unwrap()).await.unwrap();
    let mut writer = BufWriter::new(file);

    loop {
        sleep(Duration::from_secs(STATISIC_INTERVAL)).await;
        
        for (id, ctx) in all_schedule_contexts.iter().enumerate() {
            let jsonl = {
                let g = ctx.as_ref().lock().await;
                g.lmetric.to_jsonl_with_id(id)
            };
            writer.write_all(jsonl.as_bytes()).await.unwrap();
            writer.write_all(b"\n").await.unwrap();
        }

        writer.flush().await.unwrap();
    }
}
