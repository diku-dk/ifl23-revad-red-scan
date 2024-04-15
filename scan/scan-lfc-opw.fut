import "scan-adj-comp"

------------------------------------------
---  2. Scan with Lin-Fun Composition  ---
------------------------------------------

-- Scan with linear-function composition: correctness
-- ==
-- entry: fwd_Jlfc_ours rev_Jlfc_ours rev_Jlfc_comp
-- compiled input { [1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32] 
--                  [6.0f32, 7.0f32, 8.0f32, 9.0f32, 10.0f32]
--                }
-- output { [ [1.000000f32, 0.000000f32, 0.000000f32, 0.000000f32, 0.000000f32]
--          , [7.000000f32, 1.000000f32, 0.000000f32, 0.000000f32, 0.000000f32]
--          , [56.000000f32, 8.000000f32, 1.000000f32, 0.000000f32, 0.000000f32]
--          , [504.000000f32, 72.000000f32, 9.000000f32, 1.000000f32, 0.000000f32]
--          , [5040.000000f32, 720.000000f32, 90.000000f32, 10.000000f32, 1.000000f32]
--          ]
--          [ [1.000000f32, 0.000000f32, 0.000000f32, 0.000000f32, 0.000000f32]
--          , [7.000000f32, 7.000000f32, 0.000000f32, 0.000000f32, 0.000000f32]
--          , [56.000000f32, 56.000000f32, 51.000000f32, 0.000000f32, 0.000000f32]
--          , [504.000000f32, 504.000000f32, 459.000000f32, 411.000000f32, 0.000000f32]
--          , [5040.000000f32, 5040.000000f32, 4590.000000f32, 4110.000000f32, 3703.000000f32]
--          ]
--        }

-- I did it by hand for shortened inputs [1, 2] and [6, 7]
-- and it seems that the reverse mode is correct.
-- Unfortunately the forward mode seems to be incorrect and returns:
-- 
-- output { [ [1f32,  0f32,  0f32,  0f32, 0f32]
--          , [7f32,  2f32,  0f32,  0f32, 0f32]
--          , [56f32, 16f32, 10f32, 0f32, 0f32]
--          , [504f32, 144f32, 90f32, 76f32, 0f32]
--          , [5040f32, 1440f32, 900f32, 760f32, 680f32]
--          ]
--          [ [1f32, 0f32, 0f32, 0f32, 0f32]
--          , [7f32, 6f32, 0f32, 0f32, 0f32]
--          , [56f32, 48f32, 42f32, 0f32, 0f32]
--          , [504f32, 432f32, 378f32, 336f32, 0f32]
--          , [5040f32, 4320f32, 3780f32, 3360f32, 3024f32]
--          ]
--        }

def plus_tup (a1: f32, a2: f32) (b1: f32, b2: f32) = (a1+b1, a2+b2)
def zero_tup = (0f32, 0f32)

-- linear-function composition (lfc) operator;
let lfc (a0: f32, a1: f32)
        (b0: f32, b1: f32) : (f32, f32) =
    (b0 + b1*a0, a1*b1)

-- neutral element for linear-function composition
let lfc_ne = (0f32, 1f32)

def primal_lfc [n] (xs: [n](f32,f32)) =
  scan lfc lfc_ne xs

entry fwd_Jlfc_ours [n] (a: [n]f32) (b: [n]f32) =
  tabulate n (\i -> jvp primal_lfc (zip a b) (replicate n (0,0) with [i] = (1,1)))
  |> transpose |> map unzip |> unzip

entry rev_Jlfc_ours [n] (a: [n]f32) (b: [n]f32) =
  tabulate n (\i -> vjp primal_lfc (zip a b) (replicate n (0,0) with [i] = (1,1)))
  |> map unzip |> unzip

entry rev_Jlfc_comp [n] (a: [n]f32) (b: [n]f32) =
  tabulate n (\i -> scan_bar zero_tup plus_tup lfc lfc_ne (zip a b) 
                             (replicate n (0,0) with [i] = (1,1))
             )
  |> map unzip |> unzip

-- Scan with linear-function composition: performance
-- ==
-- entry: scan_lfc_comp scan_lfc_ours scan_lfc_prim
-- compiled random input {  [10000000]f32  [10000000]f32  [10000000]f32  [10000000]f32 }
-- compiled random input { [100000000]f32 [100000000]f32 [100000000]f32 [100000000]f32 }

entry scan_lfc_prim [n] (inp1 : [n]f32) 
                        (inp2 : [n]f32)
                        (_adj1 : [n]f32)
                        (_adj2 : [n]f32) : 
                        ([n]f32,[n]f32) =
  zip inp1 inp2 |> primal_lfc |> unzip

entry scan_lfc_comp [n] (inp1 : [n]f32) 
                        (inp2 : [n]f32)
                        (adj1 : [n]f32)
                        (adj2 : [n]f32) : 
                        ([n]f32,[n]f32) =
  scan_bar zero_tup plus_tup lfc lfc_ne (zip inp1 inp2) (zip adj1 adj2) 
    |> unzip

entry scan_lfc_ours [n] (inp1 : [n]f32) 
                        (inp2 : [n]f32)
                        (adj1 : [n]f32)
                        (adj2 : [n]f32) : 
                        ([n]f32,[n]f32) =
                        -- ([n]f32,[n]f32,[n]f32,[n]f32) =
  vjp primal_lfc (zip inp1 inp2) (zip adj1 adj2) |> unzip
--  let (res, inp_bar) = vjp2 primal_lfc (zip inp1 inp2) (zip adj1 adj2)
--  let (res1, res2) = unzip res
--  let (inp1_, inp2_) = unzip inp_bar
--  in  (res1, res2, inp1_, inp2_)
    
    
-- Scan with linear-function composition: performance
-- ==
-- entry: scan_opw_comp scan_opw_ours scan_opw_prim
-- compiled random input {  [10000000]f32  [10000000]f32  [10000000]f32  [10000000]f32 }
-- compiled random input { [100000000]f32 [100000000]f32 [100000000]f32 [100000000]f32 }

def opw (p1: f32, s1: f32) (p2: f32, s2: f32) = (p1 + p2 + s1*s2, s1+s2)
def opw_ne = (0f32, 0f32)

def primal_opw [n] (xs: [n](f32,f32)) =
  scan opw opw_ne xs

entry scan_opw_prim [n] (inp1 : [n]f32) 
                        (inp2 : [n]f32)
                        (_adj1 : [n]f32)
                        (_adj2 : [n]f32) : 
                        ([n]f32,[n]f32) =
  zip inp1 inp2 |> primal_opw |> unzip

entry scan_opw_comp [n] (inp1 : [n]f32) 
                        (inp2 : [n]f32)
                        (adj1 : [n]f32)
                        (adj2 : [n]f32) : 
                        ([n]f32,[n]f32) =
  scan_bar zero_tup plus_tup opw opw_ne (zip inp1 inp2) (zip adj1 adj2) 
    |> unzip

entry scan_opw_ours [n] (inp1 : [n]f32) 
                        (inp2 : [n]f32)
                        (adj1 : [n]f32)
                        (adj2 : [n]f32) : 
                        ([n]f32,[n]f32) =
                        -- ([n]f32,[n]f32,[n]f32,[n]f32) =
   vjp primal_opw (zip inp1 inp2) (zip adj1 adj2) |> unzip
--  let (res, adj) = vjp2 primal_opw (zip inp1 inp2) (zip adj1 adj2)
--  let (res1, res2) = unzip res
--  let (adj1, adj2) = unzip adj
--  in  (res1, res2, adj1, adj2)
