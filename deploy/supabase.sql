-- SafeSpace: tables for saved check-in scores. Run once in the Supabase SQL Editor.
--
-- Only scores and times are stored, never words, recordings or photos.
-- Row-level security: a signed-in person can read, add and delete only their own
-- rows; people who are not signed in can do nothing. Deleting a Supabase user
-- deletes their rows too.

create table if not exists public.consents (
  user_id uuid primary key default auth.uid() references auth.users (id) on delete cascade,
  consented_at timestamptz not null default now()
);

create table if not exists public.checkins (
  id bigint generated always as identity primary key,
  user_id uuid not null default auth.uid() references auth.users (id) on delete cascade,
  t double precision not null,          -- check-in time, seconds since 1970 (UTC)
  wrist real check (wrist between 0 and 1),
  text  real check (text  between 0 and 1),
  voice real check (voice between 0 and 1),
  face  real check (face  between 0 and 1)
);
create index if not exists checkins_user_t on public.checkins (user_id, t);

alter table public.consents enable row level security;
alter table public.checkins enable row level security;

revoke all on public.consents, public.checkins from anon;
grant select, insert, delete on public.consents, public.checkins to authenticated;

drop policy if exists "own consent: read" on public.consents;
drop policy if exists "own consent: add" on public.consents;
drop policy if exists "own consent: delete" on public.consents;
create policy "own consent: read"   on public.consents for select to authenticated using ((select auth.uid()) = user_id);
create policy "own consent: add"    on public.consents for insert to authenticated with check ((select auth.uid()) = user_id);
create policy "own consent: delete" on public.consents for delete to authenticated using ((select auth.uid()) = user_id);

drop policy if exists "own check-ins: read" on public.checkins;
drop policy if exists "own check-ins: add" on public.checkins;
drop policy if exists "own check-ins: delete" on public.checkins;
create policy "own check-ins: read"   on public.checkins for select to authenticated using ((select auth.uid()) = user_id);
create policy "own check-ins: add"    on public.checkins for insert to authenticated with check ((select auth.uid()) = user_id);
create policy "own check-ins: delete" on public.checkins for delete to authenticated using ((select auth.uid()) = user_id);
