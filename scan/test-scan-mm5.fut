import "scan-adj-comp"

-------------------------------------------------
---  5. Scan with 5x5 Matrix Multipplication  ---
-------------------------------------------------

type tp5x5f32 = ( f32,f32,f32,f32,f32
                , f32,f32,f32,f32,f32
                , f32,f32,f32,f32,f32
                , f32,f32,f32,f32,f32
                , f32,f32,f32,f32,f32
                )

let zero : tp5x5f32 = 
    ( 0,0,0,0,0
    , 0,0,0,0,0
    , 0,0,0,0,0
    , 0,0,0,0,0
    , 0,0,0,0,0
    )
    
let plus ( ( a0,  a1,  a2,  a3,  a4
           , a5,  a6,  a7,  a8,  a9
           , a10, a11, a12, a13, a14
           , a15, a16, a17, a18, a19
           , a20, a21, a22, a23, a24
           ) : tp5x5f32
         )
         ( ( b0,  b1,  b2,  b3,  b4
           , b5,  b6,  b7,  b8,  b9
           , b10, b11, b12, b13, b14
           , b15, b16, b17, b18, b19
           , b20, b21, b22, b23, b24
           ) : tp5x5f32
         ) : tp5x5f32 =
  ( a0 + b0, a1 + b1, a2 + b2, a3 + b3, a4 + b4
  , a5 + b5, a6 + b6, a7 + b7, a8 + b8, a9 + b9
  , a10+b10, a11+b11, a12+b12, a13+b13, a14+b14
  , a15+b15, a16+b16, a17+b17, a18+b18, a19+b19
  , a20+b20, a21+b21, a22+b22, a23+b23, a24+b24
  )


def mm5by5 ( ( a_00, a_01, a_02, a_03, a_04
             , a_10, a_11, a_12, a_13, a_14
             , a_20, a_21, a_22, a_23, a_24
             , a_30, a_31, a_32, a_33, a_34
             , a_40, a_41, a_42, a_43, a_44
             ) : tp5x5f32
           )
           ( ( b_00, b_01, b_02, b_03, b_04
             , b_10, b_11, b_12, b_13, b_14
             , b_20, b_21, b_22, b_23, b_24
             , b_30, b_31, b_32, b_33, b_34
             , b_40, b_41, b_42, b_43, b_44
             ) : tp5x5f32
           )
           : tp5x5f32 =
  ( -- first row:
    a_00*b_00 + a_01*b_10 + a_02*b_20 + a_03*b_30 + a_04*b_40
  , a_00*b_01 + a_01*b_11 + a_02*b_21 + a_03*b_31 + a_04*b_41
  , a_00*b_02 + a_01*b_12 + a_02*b_22 + a_03*b_32 + a_04*b_42
  , a_00*b_03 + a_01*b_13 + a_02*b_23 + a_03*b_33 + a_04*b_43
  , a_00*b_04 + a_01*b_14 + a_02*b_24 + a_03*b_34 + a_04*b_44
    -- second row:
  , a_10*b_00 + a_11*b_10 + a_12*b_20 + a_13*b_30 + a_14*b_40
  , a_10*b_01 + a_11*b_11 + a_12*b_21 + a_13*b_31 + a_14*b_41
  , a_10*b_02 + a_11*b_12 + a_12*b_22 + a_13*b_32 + a_14*b_42
  , a_10*b_03 + a_11*b_13 + a_12*b_23 + a_13*b_33 + a_14*b_43
  , a_10*b_04 + a_11*b_14 + a_12*b_24 + a_13*b_34 + a_14*b_44
    -- third row:
  , a_20*b_00 + a_21*b_10 + a_22*b_20 + a_23*b_30 + a_24*b_40
  , a_20*b_01 + a_21*b_11 + a_22*b_21 + a_23*b_31 + a_24*b_41
  , a_20*b_02 + a_21*b_12 + a_22*b_22 + a_23*b_32 + a_24*b_42
  , a_20*b_03 + a_21*b_13 + a_22*b_23 + a_23*b_33 + a_24*b_43
  , a_20*b_04 + a_21*b_14 + a_22*b_24 + a_23*b_34 + a_24*b_44    
    -- fourth row:
  , a_30*b_00 + a_31*b_10 + a_32*b_20 + a_33*b_30 + a_34*b_40
  , a_30*b_01 + a_31*b_11 + a_32*b_21 + a_33*b_31 + a_34*b_41
  , a_30*b_02 + a_31*b_12 + a_32*b_22 + a_33*b_32 + a_34*b_42
  , a_30*b_03 + a_31*b_13 + a_32*b_23 + a_33*b_33 + a_34*b_43
  , a_30*b_04 + a_31*b_14 + a_32*b_24 + a_33*b_34 + a_34*b_44    
    -- fifth row:
  , a_40*b_00 + a_41*b_10 + a_42*b_20 + a_43*b_30 + a_44*b_40
  , a_40*b_01 + a_41*b_11 + a_42*b_21 + a_43*b_31 + a_44*b_41
  , a_40*b_02 + a_41*b_12 + a_42*b_22 + a_43*b_32 + a_44*b_42
  , a_40*b_03 + a_41*b_13 + a_42*b_23 + a_43*b_33 + a_44*b_43
  , a_40*b_04 + a_41*b_14 + a_42*b_24 + a_43*b_34 + a_44*b_44    
  )


let mm5x5_ne : tp5x5f32 = 
  ( 1,0,0,0,0
  , 0,1,0,0,0
  , 0,0,1,0,0
  , 0,0,0,1,0
  , 0,0,0,0,1
  )

def primal2 [n] (xs: [n]tp5x5f32) =
  scan mm5by5 mm5x5_ne xs

def fromarrs2 = map (\(x: [25]f32) -> ( x[0], x[1], x[2], x[3], x[4]
                                      , x[5], x[6], x[7], x[8], x[9]
                                      , x[10],x[11],x[12],x[13],x[14] 
                                      , x[15],x[16],x[17],x[18],x[19]
                                      , x[20],x[21],x[22],x[23],x[24]
                                      )
                    )
def toarrs2 = map (\(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p, s,t,u,v,x,w,y,z,aa) -> 
                    [a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p, s,t,u,v,x,w,y,z,aa]
                  )

def onehot_2d n m x y =
  tabulate_2d n m (\i j -> f32.bool((i,j) == (x,y)))

entry fwd [n] (input: [n][25]f32) : [n][25][n][25]f32 =
  let input = fromarrs2 input
  in tabulate (n*25) (\i -> jvp primal2 input (fromarrs2 (onehot_2d n 25 (i/25) (i%25))))
     |> map toarrs2 |> transpose |> map transpose |> map (map unflatten)

entry rev [n] (input: [n][25]f32) : [n][25][n][25]f32 =
  let input = fromarrs2 input
  in tabulate (n*25) (\i -> vjp primal2 input (fromarrs2 (onehot_2d n 25 (i/25) (i%25))))
     |> unflatten |> map (map toarrs2)

-- Scan with 5x5 matrix multiplication: performance
-- ==
-- entry: scan_mm5_prim scan_mm5_comp scan_mm5_ours
-- compiled random input { [25][10000000]f32 [25][10000000]f32 }
-- compiled random input { [25][50000000]f32 [25][50000000]f32 }

def fromarrs5T [n] (x: [25][n]f32) = 
  map (\i -> ( x[0,i], x[1,i], x[2,i], x[3,i], x[4,i]
             , x[5,i], x[6,i], x[7,i], x[8,i], x[9,i]
             , x[10,i],x[11,i],x[12,i],x[13,i],x[14,i]
             , x[15,i],x[16,i],x[17,i],x[18,i],x[19,i]
             , x[20,i],x[21,i],x[22,i],x[23,i],x[24,i]
             )
      ) (iota n)

entry scan_mm5_prim [n] (inp : [25][n]f32) 
                        (_adj: [25][n]f32) =
  fromarrs5T inp |> primal2

entry scan_mm5_comp [n] (inp : [25][n]f32)
                        (adj : [25][n]f32) =
  scan_bar zero plus mm5by5 mm5x5_ne
            (fromarrs5T inp)
            (fromarrs5T adj)

entry scan_mm5_ours [n] (inp : [25][n]f32)
                        (adj : [25][n]f32) =
  vjp primal2 (fromarrs5T inp) (fromarrs5T adj)
