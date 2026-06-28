"""
Crystal Lookup and Verify
==========================
A small web tool that looks up structures on the Materials Project and
sanity-checks them with a machine-learning interatomic potential, the same
kind of stability check used in Objective 1.

HOW TO RUN
----------
1. Run this in the environment that has mp-api and an MLIP installed
   (your mattersim-env works, since it has both mp-api and mattersim):

       conda activate mattersim-env
       pip install flask          # if not already installed
       python crystal_lookup_app.py

2. Open the address it prints (usually http://127.0.0.1:5000) in a browser.
3. Type a chemical system (Al-Co-Ni), an exact formula (Al2CoNi), or an
   mp-id (mp-1229050), and search. Click "Verify" on any row to relax it
   with the MLIP and see the O1-style stability diagnostics.

NOTE ON THE API KEY
-------------------
Set your Materials Project API key below, or better, set it as an
environment variable MP_API_KEY so it never ends up in the file:

       export MP_API_KEY="your_key_here"
"""

import os
from flask import Flask, request, jsonify, render_template_string

# ---- configuration -------------------------------------------------------
API_KEY = os.environ.get("MP_API_KEY", "PASTE_YOUR_KEY_HERE")

app = Flask(__name__)

# lazy globals so the heavy model only loads once, on first verification
_calc = None


def get_calculator():
    """Load the MLIP once and reuse it."""
    global _calc
    if _calc is None:
        from mattersim.forcefield import MatterSimCalculator
        _calc = MatterSimCalculator(device="cpu")
    return _calc


# ---- core logic (the same pipeline proven in the notebook) ---------------
def lookup_structure(query, api_key=API_KEY):
    from mp_api.client import MPRester
    with MPRester(api_key) as mpr:
        kwargs = dict(
            fields=["material_id", "formula_pretty", "volume", "density",
                    "symmetry", "nsites", "formation_energy_per_atom"]
        )
        if query.startswith("mp-"):
            kwargs["material_ids"] = [query]
        elif "-" in query:
            kwargs["chemsys"] = query
        else:
            kwargs["formula"] = query
        docs = mpr.materials.summary.search(**kwargs)

    out = []
    for d in docs:
        fe = d.formation_energy_per_atom
        out.append({
            "mp_id": str(d.material_id),
            "formula": d.formula_pretty,
            "n_atoms": d.nsites,
            "volume": round(d.volume, 2),
            "density": round(d.density, 3),
            "spacegroup": d.symmetry.symbol if d.symmetry else "n/a",
            "formation_energy_per_atom": round(fe, 4) if fe is not None else None,
        })
    out.sort(key=lambda r: r["n_atoms"])
    return out


def verify_structure(mp_id, api_key=API_KEY):
    import numpy as np
    from mp_api.client import MPRester
    from pymatgen.io.ase import AseAtomsAdaptor
    from ase.optimize import BFGS
    from ase.filters import FrechetCellFilter

    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            material_ids=[mp_id], fields=["material_id", "formula_pretty", "structure"]
        )
    if not docs:
        return {"error": f"No structure found for {mp_id}"}

    atoms = AseAtomsAdaptor.get_atoms(docs[0].structure)
    atoms.calc = get_calculator()

    e0 = atoms.get_potential_energy()
    f0 = float(np.sqrt((atoms.get_forces() ** 2).sum(axis=1)).max())
    v0 = atoms.get_volume()
    p0 = atoms.get_positions().copy()

    BFGS(FrechetCellFilter(atoms), logfile=None).run(fmax=0.05, steps=200)

    e1 = atoms.get_potential_energy()
    n = len(atoms)
    disp = float(np.sqrt(((atoms.get_positions() - p0) ** 2).sum(axis=1)).max())
    dmat = atoms.get_all_distances(mic=True)
    np.fill_diagonal(dmat, 999)
    dmin = float(dmat.min())

    return {
        "mp_id": str(docs[0].material_id),
        "formula": docs[0].formula_pretty,
        "n_atoms": n,
        "dE_per_atom_meV": round(1000 * (e1 - e0) / n, 1),
        "init_force": round(f0, 3),
        "vol_change_pct": round(100 * (atoms.get_volume() - v0) / v0, 2),
        "max_disp_A": round(disp, 3),
        "min_dist_A": round(dmin, 3),
        "intact": "YES" if dmin > 1.8 else "NO",
    }


# ---- web routes ----------------------------------------------------------
@app.route("/")
def home():
    return render_template_string(PAGE)


@app.route("/search")
def search():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "Type a chemical system, formula, or mp-id."})
    try:
        return jsonify({"results": lookup_structure(query)})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/verify")
def verify():
    mp_id = request.args.get("mp_id", "").strip()
    if not mp_id:
        return jsonify({"error": "No mp_id given."})
    try:
        result = verify_structure(mp_id)
        clean = {k: (round(float(v), 4) if hasattr(v, "item") else v) for k, v in result.items()}
        return jsonify({"result": clean})
    except Exception as e:
        return jsonify({"error": str(e)})


# ---- the page (HTML + a little JS, no framework needed) ------------------
PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Crystal Lookup and Verify</title>
<style>
  body { font-family: -apple-system, Arial, sans-serif; max-width: 900px;
         margin: 40px auto; padding: 0 20px; color: #1a1a1a; }
  h1 { font-size: 1.5rem; }
  .sub { color: #666; margin-bottom: 24px; }
  input { padding: 10px; font-size: 1rem; width: 320px; border: 1px solid #ccc;
          border-radius: 6px; }
  button { padding: 10px 16px; font-size: 1rem; border: none; border-radius: 6px;
           background: #2c3e50; color: white; cursor: pointer; margin-left: 6px; }
  button:hover { background: #1a252f; }
  table { border-collapse: collapse; width: 100%; margin-top: 20px; font-size: 0.9rem; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #eee; }
  th { border-bottom: 2px solid #ccc; }
  .verify-btn { background: #16794a; padding: 5px 10px; font-size: 0.8rem; margin: 0; }
  .intact-yes { color: #16794a; font-weight: bold; }
  .intact-no { color: #b00; font-weight: bold; }
  .note { color: #888; font-size: 0.8rem; margin-top: 30px; }
  .err { color: #b00; margin-top: 16px; }
  .spin { color: #888; font-style: italic; }
</style>
</head>
<body>
  <h1>Crystal Lookup and Verify</h1>
  <p class="sub">Look up a structure on the Materials Project, then sanity-check it
     with a machine-learning interatomic potential (the same stability check used in
     Objective 1). Try <code>Al-Co-Ni</code>, <code>Al2CoNi</code>, or <code>mp-1229050</code>.</p>

  <input id="q" placeholder="Al-Co-Ni" onkeydown="if(event.key==='Enter')doSearch()">
  <button onclick="doSearch()">Search</button>

  <div id="out"></div>

  <p class="note">Lookup data: Materials Project API. Verification: single-stage
     relaxation with an MLIP, reporting energy change, initial force, volume change,
     maximum displacement, and minimum interatomic distance. "Intact" means the
     minimum distance stayed above the 1.8 A collapse threshold.</p>

<script>
async function doSearch() {
  const q = document.getElementById('q').value;
  const out = document.getElementById('out');
  out.innerHTML = '<p class="spin">Searching Materials Project...</p>';
  const r = await fetch('/search?q=' + encodeURIComponent(q));
  const data = await r.json();
  if (data.error) { out.innerHTML = '<p class="err">' + data.error + '</p>'; return; }
  if (!data.results.length) { out.innerHTML = '<p>No structures found.</p>'; return; }

  let html = '<table><tr><th>mp-id</th><th>formula</th><th>atoms</th>'
           + '<th>volume</th><th>spacegroup</th><th>E<sub>form</sub></th><th></th></tr>';
  for (const s of data.results) {
    html += '<tr>'
      + '<td>' + s.mp_id + '</td>'
      + '<td>' + s.formula + '</td>'
      + '<td>' + s.n_atoms + '</td>'
      + '<td>' + s.volume + '</td>'
      + '<td>' + s.spacegroup + '</td>'
      + '<td>' + (s.formation_energy_per_atom ?? '-') + '</td>'
      + '<td><button class="verify-btn" onclick="doVerify(\\'' + s.mp_id + '\\', this)">Verify</button></td>'
      + '</tr>'
      + '<tr id="v-' + s.mp_id + '"><td colspan="7"></td></tr>';
  }
  html += '</table>';
  out.innerHTML = html;
}

async function doVerify(mpId, btn) {
  const row = document.getElementById('v-' + mpId);
  row.innerHTML = '<td colspan="7" class="spin">Relaxing with MLIP, this takes a moment...</td>';
  btn.disabled = true;
  const r = await fetch('/verify?mp_id=' + encodeURIComponent(mpId));
  const data = await r.json();
  btn.disabled = false;
  if (data.error) { row.innerHTML = '<td colspan="7" class="err">' + data.error + '</td>'; return; }
  const v = data.result;
  const intact = v.intact === 'YES'
     ? '<span class="intact-yes">YES</span>' : '<span class="intact-no">NO</span>';
  row.innerHTML = '<td colspan="7">'
    + 'energy change: <b>' + v.dE_per_atom_meV + ' meV/atom</b> &nbsp; '
    + 'initial force: <b>' + v.init_force + ' eV/A</b> &nbsp; '
    + 'volume change: <b>' + v.vol_change_pct + '%</b> &nbsp; '
    + 'max displacement: <b>' + v.max_disp_A + ' A</b> &nbsp; '
    + 'min distance: <b>' + v.min_dist_A + ' A</b> &nbsp; '
    + 'intact: ' + intact
    + '</td>';
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("Starting Crystal Lookup and Verify...")
    print("Open http://127.0.0.1:5000 in your browser.")
    app.run(debug=False, port=5000)
