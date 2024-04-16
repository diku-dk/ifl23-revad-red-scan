-- ==
--  entry: primalSatAdd revadSatAdd
-- compiled input@data/histo-31-10mil.in
-- compiled input@data/histo-401-10mil.in
-- compiled input@data/histo-500K-10mil.in
-- compiled input@data/histo-31-100mil.in
-- compiled input@data/histo-401-100mil.in
-- compiled input@data/histo-500K-100mil.in

def satAdd (x: f32) (y: f32) : f32 =
  let s = x + y
  in if s > 1000000
     then 1000000
     else s

def fsatAdd [n][m] (is: [n]i64) (dst: [m]f32, vs: [n]f32) =
  reduce_by_index (copy dst) satAdd 0.0f32 is vs

entry primalSatAdd [n][m] (is: [n]i64) (dst: [m]f32) (vs: [n]f32) (_ones: [m]f32) =
  fsatAdd is (dst, vs)

entry revadSatAdd [n][m] (is: [n]i64) (dst: [m]f32) (vs: [n]f32) (ones: [m]f32) =
  let (r, (adj_dst, adj_vs)) = vjp2 (fsatAdd is) (dst, vs) ones
  in  (r, adj_dst, adj_vs)
