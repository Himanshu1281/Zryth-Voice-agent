create table agents (
  id                   uuid primary key default gen_random_uuid(),
  agent_name           text not null default 'Maya',
  company_name         text not null,
  use_case             text not null check (use_case in ('promotional','lead_generation','support','custom')),
  custom_instructions  text,                 -- free text the client adds, layered on top
  did_number           text unique,          -- which phone number routes to this config
  default_language     text not null default 'en',
  status               text not null default 'draft' check (status in ('draft','live','paused')),
  created_at           timestamptz default now()
);

alter table zryth_knowledge add column agent_id uuid references agents(id);
alter table calls           add column agent_id uuid references agents(id);

create or replace function match_knowledge(
  query_embedding vector(3072),
  match_threshold float,
  match_count int,
  p_agent_id uuid
)
returns table (id bigint, content text, metadata jsonb, similarity float)
language sql stable as $$
  select
    zryth_knowledge.id,
    zryth_knowledge.content,
    zryth_knowledge.metadata,
    1 - (zryth_knowledge.embedding <=> query_embedding) as similarity
  from zryth_knowledge
  where agent_id = p_agent_id
    and 1 - (zryth_knowledge.embedding <=> query_embedding) > match_threshold
  order by zryth_knowledge.embedding <=> query_embedding
  limit match_count;
$$;
