-- 드로잉 레퍼런스 검증 도구 DB 스키마
-- Supabase 대시보드 > SQL Editor > New query 에 전체를 붙여넣고 Run.
-- 여러 번 실행해도 안전하다 (이미 있는 테이블·정책은 건너뜀).

-- ---------- 이미지 (metadata.csv) ----------
create table if not exists public.images (
  id          text primary key,                 -- Safebooru post id
  url         text,
  source_page text,
  tags        text[] not null default '{}',     -- Safebooru 원본 태그
  width       int,
  height      int,
  phash       text,
  created_at  timestamptz not null default now()
);
create index if not exists images_tags_gin on public.images using gin (tags);

-- ---------- 팀 태그 정의 (웹앱 '태그 관리' 페이지에서 편집) ----------
create table if not exists public.tags (
  key         text primary key check (key ~ '^[a-z][a-z0-9_]*$'),  -- 영문 식별자. 만든 뒤에는 바꾸지 않는다
  name        text not null check (length(trim(name)) > 0),        -- 화면 표시명
  tag_group   text not null default '',         -- 분류 축 (앵글, 포즈, 구도 ...)
  definition  text not null default '',         -- 라벨링 기준
  booru_hint  text,                             -- 대응되는 Safebooru 태그 (선택)
  active      boolean not null default true,    -- false = 사용 안 함 (라벨이 있어 삭제할 수 없는 태그)
  sort_order  int  not null default 0,
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now()
);

-- ---------- 라벨 ----------
create table if not exists public.labels (
  id          bigint generated always as identity primary key,
  image_id    text not null references public.images(id) on delete cascade,
  tag_key     text not null references public.tags(key) on update cascade on delete restrict,  -- 라벨 있는 태그는 삭제 불가
  value       smallint not null check (value in (0, 1)),   -- 1=해당, 0=해당 아님
  labeler     text not null check (length(trim(labeler)) > 0),
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now(),
  unique (image_id, tag_key, labeler)           -- 같은 사람이 다시 저장하면 덮어쓰기
);
create index if not exists labels_tag_key_idx on public.labels (tag_key);

-- ---------- 예측 결과 ----------
create table if not exists public.predictions (
  image_id     text not null references public.images(id) on delete cascade,
  tag_key      text not null references public.tags(key) on update cascade on delete cascade,
  prob         real not null check (prob between 0 and 1),
  model        text not null,
  predicted_at timestamptz not null default now(),
  primary key (image_id, tag_key)
);

-- ---------- 테이블 사용 권한 (GRANT) ----------
-- 이 프로젝트는 SQL로 만든 테이블에 API 역할 권한을 자동으로 주지 않으므로 직접 준다.
-- (없으면 secret 키로도 "permission denied for table ..." 오류가 난다)
--   service_role  = secret 키 (관리용: 이전 스크립트, 정리 작업)
--   anon          = publishable 키 (팀원용)
grant usage on schema public to anon, authenticated, service_role;
grant select, insert, update, delete on public.images, public.tags, public.labels, public.predictions to service_role;
grant select                         on public.images                    to anon, authenticated;
grant select, insert, update, delete on public.tags                      to anon, authenticated;  -- 삭제는 아래 RLS가 '라벨 없는 태그'로 제한
grant select, insert, update         on public.labels, public.predictions to anon, authenticated;
grant usage, select on all sequences in schema public to anon, authenticated, service_role;

-- ---------- 행 단위 접근 규칙 (RLS) ----------
-- secret 키는 RLS를 무시하므로 아래 정책은 publishable(anon) 키로 접속하는 팀원에게 적용된다.
-- 팀원은 읽기 + 라벨/태그/예측 추가·수정만 가능하고, 이미지 수정과 삭제는 막는다.
-- 예외: 라벨이 하나도 없는 태그는 삭제할 수 있다 (잘못 만든 태그 정리용).
alter table public.images      enable row level security;
alter table public.tags        enable row level security;
alter table public.labels      enable row level security;
alter table public.predictions enable row level security;

do $$
declare
  t text;
begin
  -- 모든 테이블: 읽기 허용
  foreach t in array array['images', 'tags', 'labels', 'predictions'] loop
    if not exists (select 1 from pg_policies where schemaname = 'public' and tablename = t and policyname = t || '_select') then
      execute format('create policy %I on public.%I for select to anon, authenticated using (true)', t || '_select', t);
    end if;
  end loop;
  -- 태그·라벨·예측: 추가·수정 허용
  foreach t in array array['tags', 'labels', 'predictions'] loop
    if not exists (select 1 from pg_policies where schemaname = 'public' and tablename = t and policyname = t || '_insert') then
      execute format('create policy %I on public.%I for insert to anon, authenticated with check (true)', t || '_insert', t);
    end if;
    if not exists (select 1 from pg_policies where schemaname = 'public' and tablename = t and policyname = t || '_update') then
      execute format('create policy %I on public.%I for update to anon, authenticated using (true) with check (true)', t || '_update', t);
    end if;
  end loop;
  if not exists (select 1 from pg_policies where schemaname = 'public' and tablename = 'tags' and policyname = 'tags_delete_unused') then
    create policy tags_delete_unused on public.tags for delete to anon, authenticated
      using (not exists (select 1 from public.labels l where l.tag_key = tags.key));
  end if;
end $$;
