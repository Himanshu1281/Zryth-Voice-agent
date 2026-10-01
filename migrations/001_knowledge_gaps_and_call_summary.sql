-- Knowledge-gap logging + post-call summaries.
-- Run once in the Supabase SQL editor.

create table if not exists public.knowledge_gaps (
    id          uuid primary key default gen_random_uuid(),
    call_id     uuid references public.calls(id) on delete set null,
    query       text not null,
    resolved    boolean not null default false,
    created_at  timestamptz not null default now()
);

create index if not exists knowledge_gaps_unresolved_idx
    on public.knowledge_gaps (created_at desc) where not resolved;

alter table public.calls add column if not exists summary text;
alter table public.calls add column if not exists intent  text;
alter table public.calls add column if not exists outcome text;
