#!/bin/sh

futhark dataset -b --i64-bounds=0:30 -g [10000000]i64 --f32-bounds=0.5:1.5 -g [31]f32 --f32-bounds=0.8:1.1 -g [10000000]f32 --f32-bounds=1.0:1.0 -g [31]f32 > data/histo-31-10mil.in

futhark dataset -b --i64-bounds=0:400 -g [10000000]i64 --f32-bounds=0.5:1.5 -g [401]f32 --f32-bounds=0.8:1.1 -g [10000000]f32 --f32-bounds=1.0:1.0 -g [401]f32 > data/histo-401-10mil.in

futhark dataset -b --i64-bounds=0:499999 -g [10000000]i64 --f32-bounds=0.5:1.5 -g [500000]f32 --f32-bounds=0.8:1.1 -g [10000000]f32 --f32-bounds=1.0:1.0 -g [500000]f32 > data/histo-500K-10mil.in

futhark dataset -b --i64-bounds=0:30 -g [100000000]i64 --f32-bounds=0.5:1.5 -g [31]f32 --f32-bounds=0.8:1.1 -g [100000000]f32 --f32-bounds=1.0:1.0 -g [31]f32 > data/histo-31-100mil.in

futhark dataset -b --i64-bounds=0:400 -g [100000000]i64 --f32-bounds=0.5:1.5 -g [401]f32 --f32-bounds=0.8:1.1 -g [100000000]f32 --f32-bounds=1.0:1.0 -g [401]f32 > data/histo-401-100mil.in

futhark dataset -b --i64-bounds=0:499999 -g [100000000]i64 --f32-bounds=0.5:1.5 -g [500000]f32 --f32-bounds=0.8:1.1 -g [100000000]f32 --f32-bounds=1.0:1.0 -g [500000]f32 > data/histo-500K-100mil.in


futhark dataset -b --i64-bounds=0:30 -g [1000000]i64 --f32-bounds=0.5:1.5 -g [31][10]f32 --f32-bounds=0.8:1.1 -g [1000000][10]f32 --f32-bounds=1.0:1.0 -g [31][10]f32 > data/histo-31-1milx10.in

futhark dataset -b --i64-bounds=0:400 -g [1000000]i64 --f32-bounds=0.5:1.5 -g [401][10]f32 --f32-bounds=0.8:1.1 -g [1000000][10]f32 --f32-bounds=1.0:1.0 -g [401][10]f32 > data/histo-401-1milx10.in

futhark dataset -b --i64-bounds=0:49999 -g [1000000]i64 --f32-bounds=0.5:1.5 -g [50000][10]f32 --f32-bounds=0.8:1.1 -g [1000000][10]f32 --f32-bounds=1.0:1.0 -g [50000][10]f32 > data/histo-50K-1milx10.in

futhark dataset -b --i64-bounds=0:30 -g [10000000]i64 --f32-bounds=0.5:1.5 -g [31][10]f32 --f32-bounds=0.8:1.1 -g [10000000][10]f32 --f32-bounds=1.0:1.0 -g [31][10]f32 > data/histo-31-10milx10.in

futhark dataset -b --i64-bounds=0:400 -g [10000000]i64 --f32-bounds=0.5:1.5 -g [401][10]f32 --f32-bounds=0.8:1.1 -g [10000000][10]f32 --f32-bounds=1.0:1.0 -g [401][10]f32 > data/histo-401-10milx10.in

futhark dataset -b --i64-bounds=0:49999 -g [10000000]i64 --f32-bounds=0.5:1.5 -g [50000][10]f32 --f32-bounds=0.8:1.1 -g [10000000][10]f32 --f32-bounds=1.0:1.0 -g [50000][10]f32 > data/histo-50K-10milx10.in

# Generated by ChatGPT:
# Function to convert human-readable values to numeric
convert_format() {
  local value="$1"
  case "$value" in
    *K)
      echo "$(( ${value%K} * 1000 ))"
      ;;
    *M)
      echo "$(( ${value%M} * 1000000 ))"
      ;;
    *)
      echo "$value"
      ;;
  esac
}

# indices, dest1, dest2, vals1, vals2, output adjoint1, output adjoint2
gen_tupled() {
  local bins_formatted="$1"
  local values_formatted="$2"
  local bins
  local values

  bins=$(convert_format "$bins_formatted")
  values=$(convert_format "$values_formatted")
  local filename="data/histo-${bins_formatted}-${values_formatted}-tupled.in"

  futhark dataset -b --i64-bounds=0:$((bins-1)) -g [$values]i64 \
                     --f32-bounds=0.5:1.5 -g [$bins]f32 \
                     --f32-bounds=0.5:1.5 -g [$bins]f32 \
                     --f32-bounds=0.8:1.1 -g [$values]f32 \
                     --f32-bounds=0.8:1.1 -g [$values]f32 \
                     --f32-bounds=0.5:1.7 -g [$bins]f32 \
                     --f32-bounds=0.5:1.7 -g [$bins]f32 \
                     > "$filename"
  echo "Dataset generated and saved to $filename"
}

gen_tupled 31 10M
gen_tupled 401 10M
gen_tupled 500K 10M
gen_tupled 31 100M
gen_tupled 401 100M
gen_tupled 500K 100M
