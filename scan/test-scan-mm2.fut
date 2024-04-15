import "scan-adj-comp"

-------------------------------------------------
---  3. Scan with 2x2 Matrix Multipplication  ---
-------------------------------------------------

def mm2by2  (a1: f32, b1: f32, c1: f32, d1: f32)
            (a2: f32, b2: f32, c2: f32, d2: f32) =
  ( a1*a2 + b1*c2
  , a1*b2 + b1*d2
  , c1*a2 + d1*c2
  , c1*b2 + d1*d2
  )

def mm2_ne = (1f32, 0f32, 0f32, 1f32)

def primal [n] (xs: [n](f32,f32,f32,f32)) =
  scan mm2by2 mm2_ne xs

def fromarrs = map (\(x: [4]f32) -> (x[0],x[1],x[2],x[3]))
def toarrs = map (\(a,b,c,d) -> [a,b,c,d])

def onehot_2d n m x y =
  tabulate_2d n m (\i j -> f32.bool((i,j) == (x,y)))

-- Scan with 2x2 matrix multiplication: correctness
-- ==
-- entry: fwd_J rev_J rev_J_comp
-- compiled input { [[1f32,2f32,3f32,4f32], [4f32,3f32,2f32,1f32], [1f32,2f32,3f32,4f32], [4f32,3f32,2f32,1f32]] }
-- output {
-- [[[[1f32, 0f32, 0f32, 0f32], [0f32, 0f32, 0f32, 0f32], [0f32, 0f32, 0f32, 0f32], [0f32, 0f32, 0f32, 0f32]],
--   [[0f32, 1f32, 0f32, 0f32], [0f32, 0f32, 0f32, 0f32], [0f32, 0f32, 0f32, 0f32], [0f32, 0f32, 0f32, 0f32]],
--   [[0f32, 0f32, 1f32, 0f32], [0f32, 0f32, 0f32, 0f32], [0f32, 0f32, 0f32, 0f32], [0f32, 0f32, 0f32, 0f32]],
--   [[0f32, 0f32, 0f32, 1f32], [0f32, 0f32, 0f32, 0f32], [0f32, 0f32, 0f32, 0f32], [0f32, 0f32, 0f32, 0f32]]],
--  [[[4f32, 2f32, 0f32, 0f32], [1f32, 0f32, 2f32, 0f32], [0f32, 0f32, 0f32, 0f32], [0f32, 0f32, 0f32, 0f32]],
--   [[3f32, 1f32, 0f32, 0f32], [0f32, 1f32, 0f32, 2f32], [0f32, 0f32, 0f32, 0f32], [0f32, 0f32, 0f32, 0f32]],
--   [[0f32, 0f32, 4f32, 2f32], [3f32, 0f32, 4f32, 0f32], [0f32, 0f32, 0f32, 0f32], [0f32, 0f32, 0f32, 0f32]],
--   [[0f32, 0f32, 3f32, 1f32], [0f32, 3f32, 0f32, 4f32], [0f32, 0f32, 0f32, 0f32], [0f32, 0f32, 0f32, 0f32]]],
--  [[[13f32, 5f32, 0f32, 0f32], [1f32, 3f32, 2f32, 6f32], [8f32, 0f32, 5f32, 0f32], [0f32, 0f32, 0f32, 0f32]],
--   [[20f32, 8f32, 0f32, 0f32], [2f32, 4f32, 4f32, 8f32], [0f32, 8f32, 0f32, 5f32], [0f32, 0f32, 0f32, 0f32]],
--   [[0f32, 0f32, 13f32, 5f32], [3f32, 9f32, 4f32, 12f32], [20f32, 0f32, 13f32, 0f32], [0f32, 0f32, 0f32, 0f32]],
--   [[0f32, 0f32, 20f32, 8f32], [6f32, 12f32, 8f32, 16f32], [0f32, 20f32, 0f32, 13f32], [0f32, 0f32, 0f32, 0f32]]],
--  [[[92f32, 36f32, 0f32, 0f32], [8f32, 20f32, 16f32, 40f32], [32f32, 16f32, 20f32, 10f32], [23f32, 0f32, 36f32, 0f32]],
--   [[59f32, 23f32, 0f32, 0f32], [5f32, 13f32, 10f32, 26f32], [24f32, 8f32, 15f32, 5f32], [0f32, 23f32, 0f32, 36f32]],
--   [[0f32, 0f32, 92f32, 36f32], [24f32, 60f32, 32f32, 80f32], [80f32, 40f32, 52f32, 26f32], [59f32, 0f32, 92f32, 0f32]],
--   [[0f32, 0f32, 59f32, 23f32], [15f32, 39f32, 20f32, 52f32], [60f32, 20f32, 39f32, 13f32], [0f32, 59f32, 0f32, 92f32]]]]
-- }

entry fwd_J [n] (input: [n][4]f32) : [n][4][n][4]f32 =
  let input = fromarrs input
  in tabulate (n*4) (\i -> jvp primal input (fromarrs (onehot_2d n 4 (i/4) (i%4))))
     |> map toarrs |> transpose |> map transpose |> map (map unflatten)

entry rev_J [n] (input: [n][4]f32) : [n][4][n][4]f32 =
  let input = fromarrs input
  in tabulate (n*4) (\i -> vjp primal input (fromarrs (onehot_2d n 4 (i/4) (i%4))))
     |> unflatten |> map (map toarrs)

let zero = (0f32, 0f32, 0f32, 0f32)
let plus (a1: f32, a2: f32, a3: f32, a4: f32)
         (b1: f32, b2: f32, b3: f32, b4: f32) =
  (a1+b1, a2+b2, a3+b3, a4+b4)

entry rev_J_comp [n] (input: [n][4]f32) : [n][4][n][4]f32 =
  let input = fromarrs input
  in  tabulate (n*4) (\i -> scan_bar zero plus mm2by2 mm2_ne input 
                            (fromarrs (onehot_2d n 4 (i/4) (i%4))) )
     |> unflatten |> map (map toarrs)

-- Scan with 2x2 matrix multiplication: correctness
-- ==
-- entry: scan_mm2_prim scan_mm2_comp scan_mm2_ours
-- compiled random input { [10000000]f32 [10000000]f32 [10000000]f32 [10000000]f32 [10000000]f32 [10000000]f32 [10000000]f32 [10000000]f32 }
-- compiled random input { [100000000]f32 [100000000]f32 [100000000]f32 [100000000]f32 [100000000]f32 [100000000]f32 [100000000]f32 [100000000]f32 }

entry scan_mm2_prim [n] (inp1 : [n]f32) 
                        (inp2 : [n]f32)
                        (inp3 : [n]f32)
                        (inp4 : [n]f32)
                        (_adj1 : [n]f32)
                        (_adj2 : [n]f32) 
                        (_adj3 : [n]f32)
                        (_adj4 : [n]f32) : 
                        ([n]f32,[n]f32,[n]f32,[n]f32) =
  zip4 inp1 inp2 inp3 inp4 |> primal |> unzip4

entry scan_mm2_comp [n] (inp1 : [n]f32)
                        (inp2 : [n]f32)
                        (inp3 : [n]f32)
                        (inp4 : [n]f32)
                        (adj1 : [n]f32)
                        (adj2 : [n]f32)
                        (adj3 : [n]f32)
                        (adj4 : [n]f32) : 
                        ([n]f32,[n]f32,[n]f32,[n]f32) =
  scan_bar zero plus mm2by2 mm2_ne
            (zip4 inp1 inp2 inp3 inp4)
            (zip4 adj1 adj2 adj3 adj4) 
    |> unzip4

entry scan_mm2_ours [n] (inp1 : [n]f32) 
                        (inp2 : [n]f32)
                        (inp3 : [n]f32)
                        (inp4 : [n]f32)
                        (adj1 : [n]f32)
                        (adj2 : [n]f32)
                        (adj3 : [n]f32)
                        (adj4 : [n]f32) : 
                        ([n]f32,[n]f32,[n]f32,[n]f32) =
  vjp primal (zip4 inp1 inp2 inp3 inp4) (zip4 adj1 adj2 adj3 adj4)
    |> unzip4
