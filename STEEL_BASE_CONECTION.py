import math
from dataclasses import dataclass, asdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Circle, Rectangle


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Base Plate Modular v1",
    layout="wide",
)

st.title("Diseño modular de placa base de acero")
st.caption("Versión modular en cadena: datos, geometría, módulo uniaxial preliminar y base para ampliación.")


# ============================================================
# DATACLASSES
# ============================================================

@dataclass
class Loads:
    Pu_kN: float
    Mux_kNm: float
    Muy_kNm: float
    Vux_kN: float = 0.0
    Vuy_kN: float = 0.0


@dataclass
class Materials:
    Fy_plate_MPa: float
    Fu_plate_MPa: float
    fc_MPa: float
    Fy_anchor_MPa: float
    Fu_anchor_MPa: float


@dataclass
class ColumnGeometry:
    d_col_mm: float
    bf_col_mm: float


@dataclass
class BasePlateGeometry:
    B_bp_mm: float
    N_bp_mm: float
    tp_mm: float


@dataclass
class AnchorLayout:
    nbx: int
    nby: int
    edge_x_mm: float
    edge_y_mm: float
    db_mm: float
    Ab_mm2: float
    hef_mm: float


@dataclass
class PedestalGeometry:
    B_ped_mm: float
    N_ped_mm: float
    h_ped_mm: float


# ============================================================
# FUNCIONES AUXILIARES GENERALES
# ============================================================

def validate_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"'{name}' debe ser mayor que cero. Valor recibido: {value}")


def validate_nonnegative(value: float, name: str) -> None:
    if value < 0:
        raise ValueError(f"'{name}' no puede ser negativo. Valor recibido: {value}")


def validate_inputs(
    loads: Loads,
    materials: Materials,
    column: ColumnGeometry,
    base_plate: BasePlateGeometry,
    anchors: AnchorLayout,
    pedestal: PedestalGeometry,
) -> None:
    validate_positive(loads.Pu_kN, "Pu_kN")
    validate_nonnegative(abs(loads.Mux_kNm), "abs(Mux_kNm)")
    validate_nonnegative(abs(loads.Muy_kNm), "abs(Muy_kNm)")
    validate_nonnegative(abs(loads.Vux_kN), "abs(Vux_kN)")
    validate_nonnegative(abs(loads.Vuy_kN), "abs(Vuy_kN)")

    for name, value in asdict(materials).items():
        validate_positive(value, name)

    for name, value in asdict(column).items():
        validate_positive(value, name)

    for name, value in asdict(base_plate).items():
        validate_positive(value, name)

    if anchors.nbx < 2 or anchors.nby < 2:
        raise ValueError("Para disposición perimetral, nbx y nby deben ser al menos 2.")

    validate_positive(anchors.edge_x_mm, "edge_x_mm")
    validate_positive(anchors.edge_y_mm, "edge_y_mm")
    validate_positive(anchors.db_mm, "db_mm")
    validate_positive(anchors.Ab_mm2, "Ab_mm2")
    validate_positive(anchors.hef_mm, "hef_mm")

    for name, value in asdict(pedestal).items():
        validate_positive(value, name)

    if column.bf_col_mm > base_plate.B_bp_mm:
        raise ValueError(
            "El ancho de columna bf_col_mm no puede ser mayor que el ancho de placa B_bp_mm."
        )

    if column.d_col_mm > base_plate.N_bp_mm:
        raise ValueError(
            "El peralte de columna d_col_mm no puede ser mayor que el largo de placa N_bp_mm."
        )

    if base_plate.B_bp_mm > pedestal.B_ped_mm:
        raise ValueError(
            "La dimensión B de la placa no puede ser mayor que la dimensión B del pedestal."
        )

    if base_plate.N_bp_mm > pedestal.N_ped_mm:
        raise ValueError(
            "La dimensión N de la placa no puede ser mayor que la dimensión N del pedestal."
        )

    available_x = base_plate.B_bp_mm - 2.0 * anchors.edge_x_mm
    available_y = base_plate.N_bp_mm - 2.0 * anchors.edge_y_mm

    if available_x < 0:
        raise ValueError(
            "La distancia borde-eje de pernos en x es demasiado grande para la placa."
        )

    if available_y < 0:
        raise ValueError(
            "La distancia borde-eje de pernos en y es demasiado grande para la placa."
        )


def generate_bolt_coordinates(
    B_bp_mm: float,
    N_bp_mm: float,
    nbx: int,
    nby: int,
    edge_x_mm: float,
    edge_y_mm: float,
) -> pd.DataFrame:
    if nbx < 2 or nby < 2:
        raise ValueError("Para una disposición perimetral, nbx y nby deben ser al menos 2.")

    x_left = -B_bp_mm / 2.0 + edge_x_mm
    x_right = B_bp_mm / 2.0 - edge_x_mm
    y_bottom = -N_bp_mm / 2.0 + edge_y_mm
    y_top = N_bp_mm / 2.0 - edge_y_mm

    x_coords = np.linspace(x_left, x_right, nbx)
    y_coords = np.linspace(y_bottom, y_top, nby)

    bolts = []

    for x in x_coords:
        bolts.append((float(x), float(y_top)))

    for y in y_coords[-2:0:-1]:
        bolts.append((float(x_right), float(y)))

    for x in x_coords[::-1]:
        bolts.append((float(x), float(y_bottom)))

    for y in y_coords[1:-1]:
        bolts.append((float(x_left), float(y)))

    df = pd.DataFrame(bolts, columns=["x_mm", "y_mm"])
    df = df.drop_duplicates().reset_index(drop=True)
    df.insert(0, "Perno", np.arange(1, len(df) + 1))
    df["r_mm"] = np.sqrt(df["x_mm"] ** 2 + df["y_mm"] ** 2)

    return df


def build_input_summary(
    loads: Loads,
    materials: Materials,
    column: ColumnGeometry,
    base_plate: BasePlateGeometry,
    anchors: AnchorLayout,
    pedestal: PedestalGeometry,
) -> pd.DataFrame:
    rows = []

    for key, value in asdict(loads).items():
        rows.append(["Cargas", key, value])

    for key, value in asdict(materials).items():
        rows.append(["Materiales", key, value])

    for key, value in asdict(column).items():
        rows.append(["Columna", key, value])

    for key, value in asdict(base_plate).items():
        rows.append(["Placa base", key, value])

    for key, value in asdict(anchors).items():
        rows.append(["Pernos", key, value])

    for key, value in asdict(pedestal).items():
        rows.append(["Pedestal", key, value])

    return pd.DataFrame(rows, columns=["Bloque", "Variable", "Valor"])


def compute_layout_parameters(
    base_plate: BasePlateGeometry,
    anchors: AnchorLayout,
) -> dict:
    sx_mm = (base_plate.B_bp_mm - 2.0 * anchors.edge_x_mm) / (anchors.nbx - 1)
    sy_mm = (base_plate.N_bp_mm - 2.0 * anchors.edge_y_mm) / (anchors.nby - 1)
    total_bolts = 2 * anchors.nbx + 2 * anchors.nby - 4

    return {
        "A_plate_mm2": base_plate.B_bp_mm * base_plate.N_bp_mm,
        "total_bolts": total_bolts,
        "sx_mm": sx_mm,
        "sy_mm": sy_mm,
    }


# ============================================================
# MÓDULO 1 - ANÁLISIS UNIAXIAL PRELIMINAR
# ============================================================

def module1_uniaxial_preliminary(
    loads: Loads,
    base_plate: BasePlateGeometry,
    axis: str,
) -> dict:
    """
    Módulo 1:
    análisis preliminar uniaxial.

    Convención:
    - si axis = 'x', se usa Mux y la dimensión resistente N
      (flexión respecto al eje x produce variación de presión a lo largo de y)
    - si axis = 'y', se usa Muy y la dimensión resistente B
      (flexión respecto al eje y produce variación de presión a lo largo de x)
    """

    Pu_N = loads.Pu_kN * 1_000.0
    A_plate = base_plate.B_bp_mm * base_plate.N_bp_mm
    q_avg_MPa = Pu_N / A_plate

    if axis.lower() == "x":
        Mu_Nmm = loads.Mux_kNm * 1_000_000.0
        plate_dim_mm = base_plate.N_bp_mm
        axis_label = "x"
    elif axis.lower() == "y":
        Mu_Nmm = loads.Muy_kNm * 1_000_000.0
        plate_dim_mm = base_plate.B_bp_mm
        axis_label = "y"
    else:
        raise ValueError("El eje uniaxial debe ser 'x' o 'y'.")

    e_mm = Mu_Nmm / Pu_N
    kern_mm = plate_dim_mm / 6.0

    full_compression = abs(e_mm) <= kern_mm
    possible_uplift = abs(e_mm) > kern_mm

    return {
        "axis": axis_label,
        "Pu_kN": loads.Pu_kN,
        "Mu_kNm": Mu_Nmm / 1_000_000.0,
        "A_plate_mm2": A_plate,
        "q_avg_MPa": q_avg_MPa,
        "e_mm": e_mm,
        "kern_mm": kern_mm,
        "plate_dim_mm": plate_dim_mm,
        "full_compression": full_compression,
        "possible_uplift": possible_uplift,
        "e_over_kern": abs(e_mm) / kern_mm if kern_mm > 0 else float("nan"),
    }

# ============================================================
# MÓDULO 2 - COMPRESIÓN UNIAXIAL BAJO PLACA
# ============================================================

def module2_uniaxial_bearing(
    loads: Loads,
    base_plate: BasePlateGeometry,
    materials: Materials,
    axis: str,
) -> dict:
    """
    Módulo 2:
    análisis uniaxial de compresión bajo placa base.

    Convención:
    - axis = 'x'  -> se usa Mux y la variación de presión ocurre sobre N
    - axis = 'y'  -> se usa Muy y la variación de presión ocurre sobre B

    Hipótesis:
    - placa rígida
    - distribución lineal de presiones en compresión total
    - sin tracción del concreto
    - si e > L/6, se asume compresión parcial triangular

    Bearing conservador:
    q_allow_phi = phi * 0.85 * fc'
    con phi = 0.65
    """

    Pu_N = loads.Pu_kN * 1_000.0

    if axis.lower() == "x":
        Mu_Nmm = loads.Mux_kNm * 1_000_000.0
        L_mm = base_plate.N_bp_mm   # dimensión en la dirección de la flexión
        b_mm = base_plate.B_bp_mm   # ancho transversal
        axis_label = "x"
    elif axis.lower() == "y":
        Mu_Nmm = loads.Muy_kNm * 1_000_000.0
        L_mm = base_plate.B_bp_mm
        b_mm = base_plate.N_bp_mm
        axis_label = "y"
    else:
        raise ValueError("El eje uniaxial debe ser 'x' o 'y'.")

    A_mm2 = base_plate.B_bp_mm * base_plate.N_bp_mm
    e_mm = Mu_Nmm / Pu_N
    kern_mm = L_mm / 6.0

    phi_bearing = 0.65
    q_allow_phi_MPa = phi_bearing * 0.85 * materials.fc_MPa

    results = {
        "axis": axis_label,
        "Pu_kN": loads.Pu_kN,
        "Mu_kNm": Mu_Nmm / 1_000_000.0,
        "A_mm2": A_mm2,
        "L_mm": L_mm,
        "b_mm": b_mm,
        "e_mm": e_mm,
        "kern_mm": kern_mm,
        "phi_bearing": phi_bearing,
        "q_allow_phi_MPa": q_allow_phi_MPa,
    }

    # --------------------------------------------------------
    # CASO 1: COMPRESIÓN TOTAL
    # --------------------------------------------------------
    if abs(e_mm) <= kern_mm:
        q_avg_MPa = Pu_N / A_mm2

        # distribución lineal:
        # qmax = P/A * (1 + 6e/L)
        # qmin = P/A * (1 - 6e/L)
        factor = 6.0 * abs(e_mm) / L_mm
        q_max_MPa = q_avg_MPa * (1.0 + factor)
        q_min_MPa = q_avg_MPa * (1.0 - factor)

        # resultante de compresión coincide con la resultante aplicada
        # medida desde el centro de la placa
        xC_from_center_mm = e_mm

        results.update({
            "case": "full_compression",
            "q_avg_MPa": q_avg_MPa,
            "q_max_MPa": q_max_MPa,
            "q_min_MPa": q_min_MPa,
            "a_comp_mm": L_mm,
            "xC_from_center_mm": xC_from_center_mm,
            "bearing_ok": q_max_MPa <= q_allow_phi_MPa,
        })

    # --------------------------------------------------------
    # CASO 2: COMPRESIÓN PARCIAL
    # --------------------------------------------------------
    else:
        # longitud comprimida efectiva
        # a = 3*(L/2 - e) usando valor absoluto de e
        a_comp_mm = 3.0 * (L_mm / 2.0 - abs(e_mm))

        if a_comp_mm <= 0:
            raise ValueError(
                "La excentricidad es tan grande que la longitud comprimida efectiva resulta no positiva."
            )

        # presión máxima triangular:
        # qmax = 2P / (b * a)
        q_max_MPa = 2.0 * Pu_N / (b_mm * a_comp_mm)

        # qmin = 0 por levantamiento
        q_min_MPa = 0.0

        # ubicación de la resultante de compresión:
        # a/3 desde el borde comprimido
        # respecto al centro:
        toe_sign = 1.0 if e_mm >= 0 else -1.0
        xC_from_center_mm = toe_sign * (L_mm / 2.0 - a_comp_mm / 3.0)

        results.update({
            "case": "partial_compression",
            "q_avg_MPa": Pu_N / A_mm2,
            "q_max_MPa": q_max_MPa,
            "q_min_MPa": q_min_MPa,
            "a_comp_mm": a_comp_mm,
            "xC_from_center_mm": xC_from_center_mm,
            "bearing_ok": q_max_MPa <= q_allow_phi_MPa,
        })

    return results

# ============================================================
# MÓDULO 3 - TRACCIÓN UNIAXIAL EN PERNOS
# ============================================================

def module3_uniaxial_anchor_tension(
    loads: Loads,
    base_plate: BasePlateGeometry,
    anchors: AnchorLayout,
    bolt_df: pd.DataFrame,
    module2_results: dict,
) -> dict:
    """
    Módulo 3:
    cálculo preliminar de tracción en pernos para análisis uniaxial.

    Hipótesis:
    - si no hay levantamiento, T = 0
    - si hay compresión parcial, el momento lo equilibran:
        * compresión en concreto
        * tracción en la fila extrema de pernos del lado traccionado
    - la tracción total se reparte uniformemente entre los pernos
      de la fila extrema traccionada

    Convención:
    - axis = 'x' -> flexión asociada a Mux, variación sobre eje y
    - axis = 'y' -> flexión asociada a Muy, variación sobre eje x
    """

    axis = module2_results["axis"]
    Pu_N = loads.Pu_kN * 1_000.0

    if axis == "x":
        Mu_Nmm = loads.Mux_kNm * 1_000_000.0
        coord_name = "y_mm"
        extreme_coord = bolt_df["y_mm"].abs().max()
    elif axis == "y":
        Mu_Nmm = loads.Muy_kNm * 1_000_000.0
        coord_name = "x_mm"
        extreme_coord = bolt_df["x_mm"].abs().max()
    else:
        raise ValueError("El eje del Módulo 3 debe ser 'x' o 'y'.")

    # Si no hay compresión parcial, no se activa la tracción en este modelo
    if module2_results["case"] == "full_compression":
        return {
            "axis": axis,
            "case": "full_compression",
            "tension_active": False,
            "T_total_kN": 0.0,
            "n_tension_bolts": 0,
            "T_per_bolt_kN": 0.0,
            "lever_arm_mm": 0.0,
            "tension_side": "No aplica",
            "critical_bolts": [],
        }

    # --------------------------------------------------------
    # Lado traccionado
    # --------------------------------------------------------
    # Si e > 0, la compresión se va hacia el lado positivo.
    # La tracción aparece en el lado opuesto.
    e_mm = module2_results["e_mm"]

    if e_mm >= 0:
        tension_side = "negative"
        tension_coord = -extreme_coord
    else:
        tension_side = "positive"
        tension_coord = extreme_coord

    # Pernos activos en tracción = fila extrema del lado que levanta
    tol = 1e-6
    tension_bolts_df = bolt_df[np.isclose(bolt_df[coord_name], tension_coord, atol=tol)].copy()

    n_tension_bolts = len(tension_bolts_df)

    if n_tension_bolts == 0:
        raise ValueError("No se identificaron pernos en la fila extrema traccionada.")

    # --------------------------------------------------------
    # Brazo entre C y T
    # --------------------------------------------------------
    xC_from_center_mm = module2_results["xC_from_center_mm"]
    lever_arm_mm = abs(tension_coord - xC_from_center_mm)

    if lever_arm_mm <= 0:
        raise ValueError("El brazo interno entre compresión y tracción no es positivo.")

    # --------------------------------------------------------
    # Equilibrio de momentos
    # M = C*zC + T*zT
    # pero C = Pu + T  si tomamos equilibrio vertical con compresión positiva
    #
    # Para esta etapa preliminar, se usa equilibrio directo del par interno:
    # T_total = Mu / z
    #
    # Esto es una aproximación preliminar razonable para depuración
    # antes del refinamiento completo.
    # --------------------------------------------------------
    T_total_N = abs(Mu_Nmm) / lever_arm_mm
    T_per_bolt_N = T_total_N / n_tension_bolts

    critical_bolts = tension_bolts_df["Perno"].astype(int).tolist()

    return {
        "axis": axis,
        "case": "partial_compression",
        "tension_active": True,
        "T_total_kN": T_total_N / 1_000.0,
        "n_tension_bolts": n_tension_bolts,
        "T_per_bolt_kN": T_per_bolt_N / 1_000.0,
        "lever_arm_mm": lever_arm_mm,
        "tension_side": tension_side,
        "critical_bolts": critical_bolts,
        "tension_coord_mm": tension_coord,
    }

# ============================================================
# MÓDULO 4 - ESPESOR DE PLACA
# ============================================================

def module4_plate_thickness(
    base_plate: BasePlateGeometry,
    column: ColumnGeometry,
    materials: Materials,
    module2_results: dict,
) -> dict:
    """
    Módulo 4:
    cálculo preliminar del espesor mínimo de la placa base.

    Hipótesis:
    - la placa trabaja como franja en voladizo
    - se usa la presión máxima de contacto del Módulo 2 como presión gobernante
    - se toma el voladizo crítico entre ambas direcciones
    - resistencia a flexión simplificada de la placa:
          t_req = m * sqrt( 2*q / (phi*Fy) )

    Unidades:
    - q en MPa = N/mm²
    - Fy en MPa
    - m en mm
    - t_req en mm
    """

    B_mm = base_plate.B_bp_mm
    N_mm = base_plate.N_bp_mm
    tp_mm = base_plate.tp_mm

    bf_mm = column.bf_col_mm
    d_mm = column.d_col_mm

    Fy_MPa = materials.Fy_plate_MPa
    phi_flexure = 0.90

    # Voladizos geométricos libres
    mx_mm = (B_mm - bf_mm) / 2.0
    my_mm = (N_mm - d_mm) / 2.0

    if mx_mm <= 0:
        raise ValueError("El voladizo libre mx no es positivo. Revisa B y bf.")
    if my_mm <= 0:
        raise ValueError("El voladizo libre my no es positivo. Revisa N y d.")

    # Voladizo crítico
    mcrit_mm = max(mx_mm, my_mm)

    # Presión gobernante obtenida del módulo 2
    q_u_MPa = module2_results["q_max_MPa"]

    # Espesor mínimo requerido
    t_req_mm = mcrit_mm * math.sqrt((2.0 * q_u_MPa) / (phi_flexure * Fy_MPa))

    # Utilización
    thickness_ok = tp_mm >= t_req_mm
    utilization = t_req_mm / tp_mm if tp_mm > 0 else float("inf")

    return {
        "B_mm": B_mm,
        "N_mm": N_mm,
        "bf_mm": bf_mm,
        "d_mm": d_mm,
        "mx_mm": mx_mm,
        "my_mm": my_mm,
        "mcrit_mm": mcrit_mm,
        "q_u_MPa": q_u_MPa,
        "Fy_MPa": Fy_MPa,
        "phi_flexure": phi_flexure,
        "tp_input_mm": tp_mm,
        "t_req_mm": t_req_mm,
        "thickness_ok": thickness_ok,
        "utilization": utilization,
    }

# ============================================================
# FUNCIONES DE GRÁFICO
# ============================================================

def _draw_dimension(ax, p1, p2, offset=20, text="", text_offset=5,
                    color="black", lw=1.2, fontsize=9):
    x1, y1 = p1
    x2, y2 = p2

    if abs(y2 - y1) < 1e-9:
        y_dim = y1 + offset
        ax.plot([x1, x1], [y1, y_dim], color=color, linewidth=lw)
        ax.plot([x2, x2], [y2, y_dim], color=color, linewidth=lw)
        ax.annotate("", xy=(x1, y_dim), xytext=(x2, y_dim),
                    arrowprops=dict(arrowstyle="<->", color=color, linewidth=lw))
        ax.text((x1 + x2) / 2, y_dim + text_offset, text,
                ha="center", va="bottom", color=color, fontsize=fontsize)

    elif abs(x2 - x1) < 1e-9:
        x_dim = x1 + offset
        ax.plot([x1, x_dim], [y1, y1], color=color, linewidth=lw)
        ax.plot([x2, x_dim], [y2, y2], color=color, linewidth=lw)
        ax.annotate("", xy=(x_dim, y1), xytext=(x_dim, y2),
                    arrowprops=dict(arrowstyle="<->", color=color, linewidth=lw))
        ax.text(x_dim + text_offset, (y1 + y2) / 2, text,
                ha="left", va="center", color=color, fontsize=fontsize, rotation=90)


def _draw_steel_section(ax, column_plot: dict):
    section_type = column_plot.get("section_type", "W").upper()
    d = column_plot["d_mm"]
    bf = column_plot["bf_mm"]

    if section_type in ["W", "H", "I"]:
        tf = column_plot["tf_mm"]
        tw = column_plot["tw_mm"]

        top_flange = Rectangle((-bf / 2, d / 2 - tf), bf, tf,
                               fill=False, linewidth=2.0, linestyle="--", edgecolor="gray")
        bottom_flange = Rectangle((-bf / 2, -d / 2), bf, tf,
                                  fill=False, linewidth=2.0, linestyle="--", edgecolor="gray")
        web = Rectangle((-tw / 2, -d / 2 + tf), tw, d - 2 * tf,
                        fill=False, linewidth=2.0, linestyle="--", edgecolor="gray")

        ax.add_patch(top_flange)
        ax.add_patch(bottom_flange)
        ax.add_patch(web)

    else:
        rect = Rectangle((-bf / 2, -d / 2), bf, d,
                         fill=False, linewidth=2.0, linestyle="--", edgecolor="gray")
        ax.add_patch(rect)


def base_plate_layout_plot(
    base_plate_plot: dict,
    pedestal_plot: dict,
    column_plot: dict,
    bolt_df: pd.DataFrame,
    anchors: AnchorLayout,
):
    fig, ax = plt.subplots(figsize=(10, 10))

    Bp = pedestal_plot["B_mm"]
    Np = pedestal_plot["N_mm"]
    B = base_plate_plot["B_mm"]
    N = base_plate_plot["N_mm"]
    d = column_plot["d_mm"]
    bf = column_plot["bf_mm"]

    edge_x = anchors.edge_x_mm
    edge_y = anchors.edge_y_mm
    r = anchors.db_mm / 2.0

    x_ped_l = -Bp / 2
    x_ped_r = Bp / 2
    y_ped_b = -Np / 2
    y_ped_t = Np / 2

    x_pla_l = -B / 2
    x_pla_r = B / 2
    y_pla_b = -N / 2
    y_pla_t = N / 2

    x_col_l = -bf / 2
    x_col_r = bf / 2
    y_col_b = -d / 2
    y_col_t = d / 2

    x_bolt_left = x_pla_l + edge_x
    x_bolt_right = x_pla_r - edge_x
    y_bolt_bottom = y_pla_b + edge_y
    y_bolt_top = y_pla_t - edge_y

    ax.add_patch(Rectangle((x_ped_l, y_ped_b), Bp, Np,
                           fill=False, linewidth=2.0, edgecolor="saddlebrown"))
    ax.add_patch(Rectangle((x_pla_l, y_pla_b), B, N,
                           fill=False, linewidth=2.4, edgecolor="black"))

    _draw_steel_section(ax, column_plot)

    for _, row in bolt_df.iterrows():
        x = row["x_mm"]
        y = row["y_mm"]
        ax.add_patch(Circle((x, y), radius=r, fill=False, linewidth=1.8, edgecolor="blue"))
        ax.plot(x, y, "bo", markersize=3)
        ax.text(x + 6, y + 6, str(int(row["Perno"])), fontsize=9, color="blue")

    ax.plot(0, 0, marker="+", markersize=14, markeredgewidth=2.2, color="red")
    ax.axhline(0, linestyle=":", color="red", linewidth=1.0)
    ax.axvline(0, linestyle=":", color="red", linewidth=1.0)

    _draw_dimension(ax, (x_ped_l, y_ped_b), (x_ped_r, y_ped_b),
                    offset=-55, text=f"Pedestal B = {Bp:.0f} mm", color="saddlebrown")
    _draw_dimension(ax, (x_ped_l, y_ped_b), (x_ped_l, y_ped_t),
                    offset=-55, text=f"Pedestal N = {Np:.0f} mm", color="saddlebrown")
    _draw_dimension(ax, (x_pla_l, y_pla_t), (x_pla_r, y_pla_t),
                    offset=35, text=f"Placa B = {B:.0f} mm", color="black")
    _draw_dimension(ax, (x_pla_r, y_pla_b), (x_pla_r, y_pla_t),
                    offset=35, text=f"Placa N = {N:.0f} mm", color="black")
    _draw_dimension(ax, (x_col_l, y_col_t), (x_col_r, y_col_t),
                    offset=18, text=f"bf = {bf:.0f} mm", color="gray")
    _draw_dimension(ax, (x_col_r, y_col_b), (x_col_r, y_col_t),
                    offset=18, text=f"d = {d:.0f} mm", color="gray")
    _draw_dimension(ax, (x_pla_l, y_bolt_bottom), (x_bolt_left, y_bolt_bottom),
                    offset=-35, text=f"ex = {edge_x:.0f} mm", color="blue")
    _draw_dimension(ax, (x_bolt_left, y_pla_b), (x_bolt_left, y_bolt_bottom),
                    offset=-35, text=f"ey = {edge_y:.0f} mm", color="blue")

    margin = max(Bp, Np) * 0.22
    ax.set_xlim(-Bp / 2 - margin, Bp / 2 + margin)
    ax.set_ylim(-Np / 2 - margin, Np / 2 + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("Geometría de placa base")

    return fig


# ============================================================
# SIDEBAR - ENTRADA DE DATOS
# ============================================================

st.sidebar.header("Entrada de datos")

with st.sidebar.expander("Cargas", expanded=True):
    Pu_kN = st.number_input("Pu [kN]", min_value=0.001, value=1200.0, step=10.0)
    Mux_kNm = st.number_input("Mux [kN·m]", value=80.0, step=5.0)
    Muy_kNm = st.number_input("Muy [kN·m]", value=55.0, step=5.0)
    Vux_kN = st.number_input("Vux [kN]", value=40.0, step=5.0)
    Vuy_kN = st.number_input("Vuy [kN]", value=20.0, step=5.0)

with st.sidebar.expander("Materiales", expanded=True):
    Fy_plate_MPa = st.number_input("Fy placa [MPa]", min_value=0.001, value=250.0)
    Fu_plate_MPa = st.number_input("Fu placa [MPa]", min_value=0.001, value=400.0)
    fc_MPa = st.number_input("f'c [MPa]", min_value=0.001, value=28.0)
    Fy_anchor_MPa = st.number_input("Fy anclaje [MPa]", min_value=0.001, value=414.0)
    Fu_anchor_MPa = st.number_input("Fu anclaje [MPa]", min_value=0.001, value=620.0)

with st.sidebar.expander("Columna", expanded=True):
    section_type = st.selectbox("Tipo de perfil", ["W", "H", "I", "HSS", "BOX"], index=0)
    d_mm = st.number_input("d [mm]", min_value=0.001, value=300.0)
    bf_mm = st.number_input("bf [mm]", min_value=0.001, value=250.0)
    tf_mm = st.number_input("tf [mm]", min_value=0.0, value=14.0)
    tw_mm = st.number_input("tw [mm]", min_value=0.0, value=9.0)

with st.sidebar.expander("Placa base", expanded=True):
    B_bp_mm = st.number_input("B placa [mm]", min_value=0.001, value=600.0)
    N_bp_mm = st.number_input("N placa [mm]", min_value=0.001, value=600.0)
    tp_mm = st.number_input("tp [mm]", min_value=0.001, value=25.0)

with st.sidebar.expander("Pernos", expanded=True):
    nbx = st.number_input("nbx", min_value=2, value=4, step=1)
    nby = st.number_input("nby", min_value=2, value=5, step=1)
    edge_x_mm = st.number_input("edge_x [mm]", min_value=0.001, value=100.0)
    edge_y_mm = st.number_input("edge_y [mm]", min_value=0.001, value=100.0)
    db_mm = st.number_input("db [mm]", min_value=0.001, value=20.0)
    Ab_mm2 = st.number_input("Ab [mm²]", min_value=0.001, value=245.0)
    hef_mm = st.number_input("hef [mm]", min_value=0.001, value=300.0)

with st.sidebar.expander("Pedestal", expanded=True):
    B_ped_mm = st.number_input("B pedestal [mm]", min_value=0.001, value=900.0)
    N_ped_mm = st.number_input("N pedestal [mm]", min_value=0.001, value=900.0)
    h_ped_mm = st.number_input("h pedestal [mm]", min_value=0.001, value=500.0)

with st.sidebar.expander("Modo de análisis", expanded=True):
    analysis_mode = st.selectbox(
        "Selecciona el modo",
        ["Uniaxial", "Biaxial"],
        index=0
    )
    uniaxial_axis = st.selectbox(
        "Eje uniaxial",
        ["x", "y"],
        index=0
    )


# ============================================================
# ARMAR OBJETOS
# ============================================================

loads = Loads(
    Pu_kN=Pu_kN,
    Mux_kNm=Mux_kNm,
    Muy_kNm=Muy_kNm,
    Vux_kN=Vux_kN,
    Vuy_kN=Vuy_kN,
)

materials = Materials(
    Fy_plate_MPa=Fy_plate_MPa,
    Fu_plate_MPa=Fu_plate_MPa,
    fc_MPa=fc_MPa,
    Fy_anchor_MPa=Fy_anchor_MPa,
    Fu_anchor_MPa=Fu_anchor_MPa,
)

column = ColumnGeometry(
    d_col_mm=d_mm,
    bf_col_mm=bf_mm,
)

column_plot = {
    "section_type": section_type,
    "d_mm": d_mm,
    "bf_mm": bf_mm,
    "tf_mm": tf_mm,
    "tw_mm": tw_mm,
}

base_plate = BasePlateGeometry(
    B_bp_mm=B_bp_mm,
    N_bp_mm=N_bp_mm,
    tp_mm=tp_mm,
)

anchors = AnchorLayout(
    nbx=int(nbx),
    nby=int(nby),
    edge_x_mm=edge_x_mm,
    edge_y_mm=edge_y_mm,
    db_mm=db_mm,
    Ab_mm2=Ab_mm2,
    hef_mm=hef_mm,
)

pedestal = PedestalGeometry(
    B_ped_mm=B_ped_mm,
    N_ped_mm=N_ped_mm,
    h_ped_mm=h_ped_mm,
)


# ============================================================
# EJECUCIÓN BASE EN CADENA
# ============================================================

try:
    validate_inputs(loads, materials, column, base_plate, anchors, pedestal)

    bolt_df = generate_bolt_coordinates(
        B_bp_mm=base_plate.B_bp_mm,
        N_bp_mm=base_plate.N_bp_mm,
        nbx=anchors.nbx,
        nby=anchors.nby,
        edge_x_mm=anchors.edge_x_mm,
        edge_y_mm=anchors.edge_y_mm,
    )

    summary_df = build_input_summary(
        loads=loads,
        materials=materials,
        column=column,
        base_plate=base_plate,
        anchors=anchors,
        pedestal=pedestal,
    )

    layout_info = compute_layout_parameters(base_plate, anchors)

    base_plate_plot = {
        "B_mm": base_plate.B_bp_mm,
        "N_mm": base_plate.N_bp_mm,
        "tp_mm": base_plate.tp_mm,
    }

    pedestal_plot = {
        "B_mm": pedestal.B_ped_mm,
        "N_mm": pedestal.N_ped_mm,
        "h_mm": pedestal.h_ped_mm,
    }

    # --------------------------------------------------------
    # MÓDULO 1
    # --------------------------------------------------------
    if analysis_mode == "Uniaxial":
        module1_results = module1_uniaxial_preliminary(
            loads=loads,
            base_plate=base_plate,
            axis=uniaxial_axis,
        )
    else:
        module1_results = None
    # --------------------------------------------------------
    # MÓDULO 2
    # --------------------------------------------------------
    if analysis_mode == "Uniaxial":
        module2_results = module2_uniaxial_bearing(
            loads=loads,
            base_plate=base_plate,
            materials=materials,
            axis=uniaxial_axis,
        )
    else:
        module2_results = None

    # --------------------------------------------------------
    # MÓDULO 3
    # --------------------------------------------------------
    if analysis_mode == "Uniaxial":
        module3_results = module3_uniaxial_anchor_tension(
            loads=loads,
            base_plate=base_plate,
            anchors=anchors,
            bolt_df=bolt_df,
            module2_results=module2_results,
        )
    else:
        module3_results = None   

    # --------------------------------------------------------
    # MÓDULO 4
    # --------------------------------------------------------
    if analysis_mode == "Uniaxial":
        module4_results = module4_plate_thickness(
            base_plate=base_plate,
            column=column,
            materials=materials,
            module2_results=module2_results,
        )
    else:
        module4_results = None

    # --------------------------------------------------------
    # PESTAÑAS
    # --------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Datos y geometría",
    "Módulo 1 - Uniaxial",
    "Módulo 2 - Compresión",
    "Módulo 3 - Pernos",
    "Módulo 4 - Espesor",
    "Biaxial",
    "Resumen",
    ])

    with tab1:
        st.subheader("Resumen de datos de entrada")
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        st.subheader("Parámetros geométricos")
        layout_df = pd.DataFrame(
            {"Parámetro": list(layout_info.keys()), "Valor": list(layout_info.values())}
        )
        st.dataframe(layout_df, use_container_width=True, hide_index=True)

        st.subheader("Coordenadas de pernos")
        st.dataframe(bolt_df, use_container_width=True, hide_index=True)

        st.subheader("Gráfico geométrico")
        fig = base_plate_layout_plot(
            base_plate_plot=base_plate_plot,
            pedestal_plot=pedestal_plot,
            column_plot=column_plot,
            bolt_df=bolt_df,
            anchors=anchors,
        )
        st.pyplot(fig, clear_figure=True)

    with tab2:
        st.subheader("Módulo 1 - análisis uniaxial preliminar")

        if analysis_mode != "Uniaxial":
            st.info("En la barra lateral elegiste modo Biaxial. Cambia a Uniaxial para activar este módulo.")
        else:
            mod1_df = pd.DataFrame({
                "Parámetro": [
                    "Eje analizado",
                    "Pu [kN]",
                    "Mu [kN·m]",
                    "Área de placa [mm²]",
                    "Presión media q [MPa]",
                    "Excentricidad e [mm]",
                    "Núcleo central (dim/6) [mm]",
                    "Relación e/(dim/6)",
                    "Compresión total preliminar",
                    "Posible levantamiento preliminar",
                ],
                "Valor": [
                    module1_results["axis"],
                    f"{module1_results['Pu_kN']:.3f}",
                    f"{module1_results['Mu_kNm']:.3f}",
                    f"{module1_results['A_plate_mm2']:.3f}",
                    f"{module1_results['q_avg_MPa']:.5f}",
                    f"{module1_results['e_mm']:.3f}",
                    f"{module1_results['kern_mm']:.3f}",
                    f"{module1_results['e_over_kern']:.3f}",
                    "Sí" if module1_results["full_compression"] else "No",
                    "Sí" if module1_results["possible_uplift"] else "No",
                ],
            })
            st.dataframe(mod1_df, use_container_width=True, hide_index=True)

            if module1_results["possible_uplift"]:
                st.warning(
                    "La excentricidad supera el núcleo central en este análisis preliminar. "
                    "En los siguientes módulos deberá analizarse compresión parcial y tracción en pernos."
                )
            else:
                st.success(
                    "La resultante cae dentro del núcleo central en este análisis preliminar."
                )

    with tab3:
        st.subheader("Módulo 2 - compresión uniaxial bajo placa")

        if analysis_mode != "Uniaxial":
            st.info("En la barra lateral elegiste modo Biaxial. Cambia a Uniaxial para activar este módulo.")
        else:
            mod2_df = pd.DataFrame({
                "Parámetro": [
                    "Eje analizado",
                    "Pu [kN]",
                    "Mu [kN·m]",
                    "Dimensión resistente L [mm]",
                    "Ancho transversal b [mm]",
                    "Área total de placa [mm²]",
                    "Excentricidad e [mm]",
                    "Núcleo central L/6 [mm]",
                    "Caso de compresión",
                    "Presión media q_prom [MPa]",
                    "Presión máxima q_max [MPa]",
                    "Presión mínima q_min [MPa]",
                    "Longitud comprimida a [mm]",
                    "Resultante C desde centro [mm]",
                    "φ bearing",
                    "φ·0.85·f'c [MPa]",
                    "Chequeo preliminar de bearing",
                ],
                "Valor": [
                    module2_results["axis"],
                    f"{module2_results['Pu_kN']:.3f}",
                    f"{module2_results['Mu_kNm']:.3f}",
                    f"{module2_results['L_mm']:.3f}",
                    f"{module2_results['b_mm']:.3f}",
                    f"{module2_results['A_mm2']:.3f}",
                    f"{module2_results['e_mm']:.3f}",
                    f"{module2_results['kern_mm']:.3f}",
                    "Compresión total" if module2_results["case"] == "full_compression" else "Compresión parcial",
                    f"{module2_results['q_avg_MPa']:.5f}",
                    f"{module2_results['q_max_MPa']:.5f}",
                    f"{module2_results['q_min_MPa']:.5f}",
                    f"{module2_results['a_comp_mm']:.3f}",
                    f"{module2_results['xC_from_center_mm']:.3f}",
                    f"{module2_results['phi_bearing']:.3f}",
                    f"{module2_results['q_allow_phi_MPa']:.5f}",
                    "Cumple" if module2_results["bearing_ok"] else "No cumple",
                ],
            })
            st.dataframe(mod2_df, use_container_width=True, hide_index=True)

            if module2_results["case"] == "full_compression":
                st.success(
                    "La placa permanece totalmente comprimida en el análisis uniaxial simplificado."
                )
            else:
                st.warning(
                    "Se detecta compresión parcial en el análisis uniaxial simplificado. "
                    "En el siguiente módulo se deberá calcular la tracción en pernos."
                )

            if module2_results["bearing_ok"]:
                st.success(
                    "El chequeo preliminar de bearing con q_max ≤ φ·0.85·f'c cumple."
                )
            else:
                st.error(
                    "El chequeo preliminar de bearing con q_max ≤ φ·0.85·f'c no cumple."
                )    

    
    with tab4:
        st.subheader("Módulo 3 - tracción uniaxial en pernos")

        if analysis_mode != "Uniaxial":
            st.info("En la barra lateral elegiste modo Biaxial. Cambia a Uniaxial para activar este módulo.")
        else:
            mod3_df = pd.DataFrame({
                "Parámetro": [
                    "Eje analizado",
                    "Caso de compresión",
                    "Tracción activa en pernos",
                    "Lado traccionado",
                    "Brazo interno C-T [mm]",
                    "Número de pernos traccionados",
                    "Tracción total T [kN]",
                    "Tracción por perno [kN]",
                    "Fila extrema de pernos [coord. mm]",
                    "Pernos críticos",
                ],
                "Valor": [
                    module3_results["axis"],
                    "Compresión total" if module3_results["case"] == "full_compression" else "Compresión parcial",
                    "Sí" if module3_results["tension_active"] else "No",
                    module3_results["tension_side"],
                    f"{module3_results['lever_arm_mm']:.3f}",
                    f"{module3_results['n_tension_bolts']}",
                    f"{module3_results['T_total_kN']:.5f}",
                    f"{module3_results['T_per_bolt_kN']:.5f}",
                    f"{module3_results.get('tension_coord_mm', 0.0):.3f}",
                    str(module3_results["critical_bolts"]),
                ],
            })
            st.dataframe(mod3_df, use_container_width=True, hide_index=True)

            if not module3_results["tension_active"]:
                st.success(
                    "En este modelo preliminar uniaxial no se activa tracción en pernos."
                )
            else:
                st.warning(
                    "Se activa tracción en la fila extrema de pernos del lado que levanta. "
                    "El siguiente módulo verificará el espesor de placa."
                )
    
    with tab5:
        st.subheader("Módulo 4 - espesor mínimo de placa")

        if analysis_mode != "Uniaxial":
            st.info("En la barra lateral elegiste modo Biaxial. Cambia a Uniaxial para activar este módulo.")
        else:
            mod4_df = pd.DataFrame({
                "Parámetro": [
                    "B placa [mm]",
                    "N placa [mm]",
                    "bf columna [mm]",
                    "d columna [mm]",
                    "Voladizo mx [mm]",
                    "Voladizo my [mm]",
                    "Voladizo crítico m [mm]",
                    "Presión gobernante q_u [MPa]",
                    "Fy placa [MPa]",
                    "φ flexión",
                    "Espesor ingresado tp [mm]",
                    "Espesor requerido t_req [mm]",
                    "Utilización t_req/tp",
                    "Chequeo de espesor",
                ],
                "Valor": [
                    f"{module4_results['B_mm']:.3f}",
                    f"{module4_results['N_mm']:.3f}",
                    f"{module4_results['bf_mm']:.3f}",
                    f"{module4_results['d_mm']:.3f}",
                    f"{module4_results['mx_mm']:.3f}",
                    f"{module4_results['my_mm']:.3f}",
                    f"{module4_results['mcrit_mm']:.3f}",
                    f"{module4_results['q_u_MPa']:.5f}",
                    f"{module4_results['Fy_MPa']:.3f}",
                    f"{module4_results['phi_flexure']:.3f}",
                    f"{module4_results['tp_input_mm']:.3f}",
                    f"{module4_results['t_req_mm']:.3f}",
                    f"{module4_results['utilization']:.3f}",
                    "Cumple" if module4_results["thickness_ok"] else "No cumple",
                ],
            })
            st.dataframe(mod4_df, use_container_width=True, hide_index=True)

            if module4_results["thickness_ok"]:
                st.success(
                    "El espesor ingresado de la placa cumple con el espesor mínimo requerido en este modelo simplificado."
                )
            else:
                st.error(
                    "El espesor ingresado de la placa no cumple. Debe aumentarse el espesor o modificarse la geometría."
                )
    

    with tab6:
        st.subheader("Modo biaxial")
        st.info(
            "Aquí irá el módulo biaxial una vez que cerremos y depuremos bien el flujo uniaxial."
        )

    with tab7:
        st.subheader("Estado del desarrollo")
        st.write("**Módulos activos:**")
        st.write("- Base geométrica")
        st.write("- Módulo 1 uniaxial preliminar")
        st.write("- Módulo 2 compresión bajo placa")
        st.write("- Módulo 3 tracción preliminar en pernos")
        st.write("- Módulo 4 espesor de placa")
        st.write("**Siguiente módulo a integrar:**")
        st.write("- Módulo 5: acero del perno (tracción y cortante)")

except Exception as exc:
    st.error(f"Error en los datos de entrada: {exc}")