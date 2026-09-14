#!/usr/bin/env bash
# Fetch the real PS#01 input products for the North Indian Ocean.
#
#   ./scripts/download_real_data.sh 2021-01-01 2023-12-31
#
# Prerequisites (one time):
#   pip install copernicusmarine earthaccess
#   copernicusmarine login
#   python -c "import earthaccess; earthaccess.login(persist=True)"
#
# Volumes for a three-year North Indian Ocean subset, approximately:
#   OSTIA SST   6 GB    SMOS/SMAP SSS  1 GB    DUACS SLA   0.4 GB
#   OSCAR       0.5 GB  CCMP winds     2 GB    GLORYS T    70 GB
set -euo pipefail

START="${1:-2021-01-01}"
END="${2:-2023-12-31}"
OUT="${3:-data/raw}"
LON_MIN=45 LON_MAX=105 LAT_MIN=5 LAT_MAX=30

mkdir -p "$OUT"
echo "North Indian Ocean ${LAT_MIN}-${LAT_MAX}N ${LON_MIN}-${LON_MAX}E, ${START} .. ${END}"

cm_subset () {   # dataset_id  variables...  -> one NetCDF
  local id="$1"; shift
  local vars=(); for v in "$@"; do vars+=(--variable "$v"); done
  echo ">> $id"
  copernicusmarine subset --dataset-id "$id" "${vars[@]}" \
    --minimum-longitude $LON_MIN --maximum-longitude $LON_MAX \
    --minimum-latitude $LAT_MIN  --maximum-latitude $LAT_MAX \
    --start-datetime "${START}T00:00:00" --end-datetime "${END}T23:59:59" \
    --output-directory "$OUT" --output-filename "${id//\//_}.nc" \
    --force-download
}

# --- Copernicus Marine: SST, SSS, SLA -------------------------------------
cm_subset METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2                       analysed_sst
cm_subset cmems_obs-mob_glo_phy-sss_nrt_multi_P1D                   sos
cm_subset cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D   sla adt

# --- Copernicus Marine: GLORYS target, upper 1000 m only ------------------
echo ">> GLORYS12V1 temperature (this is the large one)"
copernicusmarine subset --dataset-id cmems_mod_glo_phy_my_0.083deg_P1D-m \
  --variable thetao \
  --minimum-longitude $LON_MIN --maximum-longitude $LON_MAX \
  --minimum-latitude $LAT_MIN  --maximum-latitude $LAT_MAX \
  --minimum-depth 0 --maximum-depth 1100 \
  --start-datetime "${START}T00:00:00" --end-datetime "${END}T23:59:59" \
  --output-directory "$OUT" --output-filename "glorys_thetao.nc" --force-download

# --- NASA PO.DAAC: OSCAR currents and CCMP winds --------------------------
python - "$START" "$END" "$OUT" <<'PY'
import sys, earthaccess
start, end, out = sys.argv[1], sys.argv[2], sys.argv[3]
earthaccess.login()
for short_name in ("OSCAR_L4_OC_FINAL_V2.0", "CCMP_WINDS_10M6HR_L4_V3.1"):
    g = earthaccess.search_data(short_name=short_name, temporal=(start, end),
                                bounding_box=(45, 5, 105, 30))
    print(f">> {short_name}: {len(g)} granules")
    earthaccess.download(g, out)
PY

echo
echo "Done. Now harmonise and build the training arrays:"
echo "  python -m oceanembed.cli build -o data.source=cmems -o data.start=$START -o data.end=$END"
