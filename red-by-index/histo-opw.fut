-- ==
-- entry: primalOpw revadOpw revadOpwOpt
-- compiled input@data/histo-31-10M-tupled.in
-- compiled input@data/histo-401-10M-tupled.in
-- compiled input@data/histo-500K-10M-tupled.in
-- compiled input@data/histo-31-100M-tupled.in
-- compiled input@data/histo-401-100M-tupled.in
-- compiled input@data/histo-500K-100M-tupled.in

def opw (p1: f32, s1: f32) (p2: f32, s2: f32) = (p1 + p2 + s1*s2, s1+s2)
def opw_inv (acc_p: f32, acc_s: f32) (p1: f32, s1: f32) =
  let s2 = acc_s - s1
  let p2 = acc_p - p1 - s1*s2
  in  (p2, s2)

def opw_ne = (0f32, 0f32)
def opw_lft (w: f32) = (0f32, w)

def fopw [n][m] (is: [n]i64) (dst: [m](f32,f32), vs: [n](f32,f32)) =
  reduce_by_index (copy dst) opw opw_ne is vs

entry primalOpw [n][m] (is: [n]i64) (dst1: [m]f32) dst2
                       (vs1: [n]f32) vs2 (_: [m]f32) (_: [m]f32) =
  let dst = zip dst1 dst2
  let vs = zip vs1 vs2
  in fopw is (dst, vs) |> unzip

entry revadOpw [n][m] (is: [n]i64) (dst1: [m]f32) dst2
                      (vs1: [n]f32) vs2 (adj_out1: [m]f32) adj_out2 =
  let dst = zip dst1 dst2
  let vs = zip vs1 vs2
  let adj_out = zip adj_out1 adj_out2
  let (r, (adj_dst, adj_vs)) = vjp2 (fopw is) (dst, vs) adj_out
  let (r1, r2) = unzip r
  let (adj_dst1, adj_dst2) = unzip adj_dst
  let (adj_vs1, adj_vs2)   = unzip adj_vs
  in  (r1, r2, adj_dst1, adj_dst2, adj_vs1, adj_vs2)

entry revadOpwOpt [n][m] (ks: [n]i64) (dst1: [m]f32) dst2
                         (vs1: [n]f32) vs2 (adj_out1: [m]f32) adj_out2 =
  let dst = zip dst1 dst2
  let vs = zip vs1 vs2
  let adj_out = zip adj_out1 adj_out2
  -- Primal trace:
  -- let hist = reduce_by_index (copy dst) opw opw_ne ks vs
  let xs = reduce_by_index (replicate m opw_ne) opw opw_ne ks vs
  let hist = map2 opw xs dst
  -- Return sweep:
  let hist_bar = adj_out

  let f (_, s_b) (p_hist_bar, s_hist_bar) _ : (f32,f32)=
      (p_hist_bar, s_hist_bar + s_b * p_hist_bar)

  let vs_bar =
        map2 (\ (k : i64)  (ps : (f32,f32)) ->
               let (p, s) = ps
               let (p_b, s_b) = opw_inv hist[k] (p,s)
               in  f (p_b, s_b) hist_bar[k] (p, s)
             ) ks vs

  let dst_bar =
         map3 (\x d h_bar ->
                let d_bar = vjp (opw x) d h_bar
                in d_bar
              ) xs dst hist_bar

  let (r1, r2) = unzip hist
  let (adj_dst1, adj_dst2) = unzip dst_bar
  let (adj_vs1, adj_vs2) = unzip vs_bar
  in (r1, r2, adj_dst1, adj_dst2, adj_vs1, adj_vs2)
