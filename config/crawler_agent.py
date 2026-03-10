import modal
from config.general import VOLUME_PATH

START_URL = "https://muia.dia.fi.upm.es/es/"
MAX_DEPTH = 3
MAX_LINKS_PER_PAGE = 50
ALLOWED_URL_HOST_TO_CATEGORY = {
    "www.upm.es": "university",
    "www.fi.upm.es": "faculty",
    "dia.fi.upm.es": "department",
    "muia.dia.fi.upm.es": "master"
}
EXCLUDED_URLS = [
    "https://www.upm.es/Estudiantes/Atencion/SeguroAsistencia", # contentless
    "https://www.upm.es/helios/", # requires login
    "https://www.upm.es/intranet", # requires login
    "https://www.upm.es/Portal_inv", # requires login
    "https://www.upm.es/politecnica_virtual/", # requires login
    "https://www.upm.es/UPM/PRTR", # low relevance
    "https://www.upm.es/AvisoLegal", # low relevance
    "https://www.upm.es/e-politecnica/?p=11803", # low relevance
    "http://www.upm.es/UPM/Actividades_Culturales", # low relevance
    "https://www.upm.es/Estudiantes/CompromisoSocial/CooperacionDesarrollo", # low relevance
    "https://www.fi.upm.es/docs/estructura/989_Resolucion-CU-Acred-Institucional-18-05-2020.pdf", # low relevance
    "https://www.upm.es/UPM/PoliticasIgualdad", # low/mid relevance
    "https://www.upm.es/Personal", # news-type (low/mid relevance)
    "https://www.upm.es/Estudiantes", # news-type (low/mid relevance)
    "https://www.upm.es/internacional", # news-type (low/mid relevance)
    "https://www.upm.es/Investigacion", # news-type (low/mid relevance)
    "http://www.upm.es/e-politecnica/", # news-type (low/mid relevance)
    "https://www.upm.es/UPM/SalaPrensa", # news-type (low/mid relevance)
    "https://www.upm.es/FuturosEstudiantes", # news-type (low/mid relevance)
    "https://www.upm.es/UPM/SalaPrensa/epolitecnica_inv/ultimo", # news-type (low/mid relevance)
    "https://www.upm.es/?id=CON26062&prefmt=articulo&fmt=detail", # news-type (low/mid relevance)
    "https://www.upm.es/?id=CON25356&prefmt=articulo&fmt=detail", # news-type (low/mid relevance)
    "https://www.upm.es/?id=CON26077&prefmt=articulo&fmt=detail", # news-type (low/mid relevance)
    "https://www.upm.es/Estudiantes/BecasAyudasPremios/Premios", # awards/news-type (low/mid relevance)
    "https://www.upm.es/FuturosEstudiantes?id=CON23992&prefmt=articulo&fmt=detail", # news-type (low/mid relevance)
    "https://www.upm.es/?id=CON26046&prefmt=articulo&fmt=detail", # news-type (low/mid relevance)
    "https://www.upm.es/gsr/correo_alumnos/solicitud.upm", # form
    "https://www.upm.es/Estudiantes/CompeticionesEstudiantes", # outdated
    "https://www.upm.es/Estudiantes/e-EdU", # low relevance/several levels of indirection
    "https://www.upm.es/UPM/Accion_Social", # low relevance/several levels of indirection
    "https://www.upm.es/Estudiantes/CompromisoSocial", # low relevance/several levels of indirection
    "http://www.upm.es/UPM/CompromisoSocial/CooperacionDesarrollo", # low relevance/several levels of indirection
    "https://www.upm.es/Estudiantes/Estudios_Titulaciones/TitulosPropios", # Títulos propios (non-official)
    "https://www.upm.es/Estudiantes/Estudios_Titulaciones/programasAcademicos", # Bachelor and other Masters
    "https://www.upm.es/Estudiantes/OrdenacionAcademica/ActividadesAcreditablesEstudiosGrado", # Bachelor-only
    "https://www.upm.es/Estudiantes/Estudios_Titulaciones/EstudiosOficialesGrado", # Bachelor-only
    "https://www.upm.es/Estudiantes/Estudios_Titulaciones/Estudios_Master/Programas", # all Master programs
    "https://www.upm.es/Estudiantes/Estudios_Titulaciones/Estudios_Master/?fmt=detail&id=CON07704", # other Master programs
    "https://www.upm.es/Estudiantes/Estudios_Titulaciones/Estudios_Master/Matricula?prefmt=articulo&fmt=detail&id=CON07704", # other Master programs
    "https://www.upm.es/Estudiantes/BecasAyudasPremios/AyudasViajeCooperacion", # discontinued?

    # "http://www.fi.upm.es/docs/estructura/598_Informe-final-Acreditacion-Instituciona-06-03-20.pdf",
    "https://dia.fi.upm.es/pedro-larranaga-galardon-upm-2026/", # news-type (mid relevance)
    "https://dia.fi.upm.es/masteria/sites/default/files/master/Calidad/IF_EUROInf_4311905_UPM_MUIA.pdf", # contentless page
    "https://dia.fi.upm.es/masteria/sites/default/files/master/Complementos-formativos/6%20ECTS%20CF.pdf", # contentless page
    "https://dia.fi.upm.es/masteria/sites/default/files/master/General/Competencias%20por%20materias.pdf", # contentless page
    "https://dia.fi.upm.es/masteria/sites/default/files/master/Complementos-formativos/12%20ECTS%20CF.pdf", # contentless page
    "https://dia.fi.upm.es/masteria/sites/default/files/master/Calidad/Informe%20favorable%20acreditaci%C3%B3n%202010.pdf", # contentless page
    "https://dia.fi.upm.es/masteria/sites/default/files/master/Calidad/Concesi%C3%B3n%20definitiva%20sello%20EURO-INF%202017.pdf", # contentless page
    "https://dia.fi.upm.es/masteria/sites/default/files/master/Calidad/Informe%20renovaci%C3%B3n%20acreditaci%C3%B3n%202016.pdf", # contentless page
    "https://dia.fi.upm.es/la-catedratica-del-dia-asuncion-gomez-perez-ingresa-en-la-rae-para-ocupar-la-silla-q/", # news-type (mid relevance)
    "https://dia.fi.upm.es/el-dia-contribuye-al-plan-de-trabajo-europeo-para-medios-digitales-y-bienestar-humano/", # news-type (mid relevance)
    "https://dia.fi.upm.es/manuel-hermenegildo-alcanza-el-mayor-reconocimiento-internacional-en-el-ambito-de-la-ingenieria-informatica/", # news-type (mid relevance)

    "https://muia.dia.fi.upm.es/es/tag/investigador-del-programa-beatriz-galindo/", # contentless page
    "https://muia.dia.fi.upm.es/es/topdia/", # news-type (mid relevance)
    "https://muia.dia.fi.upm.es/es/category/noticias-es/", # news-type (mid relevance)
    "https://muia.dia.fi.upm.es/es/category/noticias-es/noticias-destacadas/", # news-type (mid relevance)
    "https://muia.dia.fi.upm.es/es/category/noticias-es/actualidad-es/", # news-type (mid relevance)
    "https://muia.dia.fi.upm.es/es/pedro-larranaga-nuevo-miembro-jakiunde/", # news-type (mid relevance)
    "https://muia.dia.fi.upm.es/es/investigadores-del-muia-entre-los-mas-influyentes-del-mundo/", # news-type (mid relevance)
    "https://muia.dia.fi.upm.es/es/victor-maojo-nombrado-academico-de-la-real-academia-nacional-de-medicina-de-espana/", # news-type (mid relevance)
    "https://muia.dia.fi.upm.es/es/asuncion-gomez-perez-elegida-para-ocupar-la-silla-q-de-la-rae/", # news-type (mid relevance)
    "https://muia.dia.fi.upm.es/es/iacovid19-inteligencia-artificial-para-analizar-los-articulos-cientificos-relacionados-con-el-covid-19/", # news-type (mid relevance)
    "https://muia.dia.fi.upm.es/es/concha-bielza-imparte-charla-plenaria-de-inauguracion-del-workshop-artificial-intelligence-for-the-fight-against-covid-19/", # news-type (mid relevance)
    "https://muia.dia.fi.upm.es/es/el-computational-intelligence-group-obtiene-una-de-las-5-ayudas-a-equipos-de-investigacion-cientifica-en-big-data-de-la-fundacion-bbva/", # news-type (mid relevance)
    "https://muia.dia.fi.upm.es/es/los-profesores-del-departamento-de-inteligencia-artificial-de-la-upm-concha-bielza-y-pedro-larranaga-lideraran-la-unidad-ellis-madrid-recientemente-creada/", # news-type (mid relevance)
    "https://muia.dia.fi.upm.es/wp-content/uploads/sites/6/2022/09/solicitudTitulo_ES.docx" # download triggered
]
LINKS_ONLY_URLS = [
    "http://www.fi.upm.es/",
    "http://www.fi.upm.es/?pagina=20",
    "https://www.upm.es/Estudiantes/Empleo",
    "https://www.upm.es/UPM/ServiciosTecnologicos",
    "https://www.upm.es/Estudiantes/BecasAyudasPremios",
    "http://www.upm.es/UPM/Deportes",
    "https://www.upm.es/Estudiantes/Estudios_Titulaciones",
    "https://www.upm.es/Estudiantes/NormativaLegislacion",
    "https://www.upm.es/Estudiantes/NormativaLegislacion/NormasEspecificas",
    "https://www.upm.es/Estudiantes/e-EdU/OpenCourseWare",
    "https://www.upm.es/Estudiantes/BecasAyudasPremios/Becas",
    "https://www.upm.es/Estudiantes/BecasAyudasPremios/Becas/OtroTipoBecas",
    "https://www.upm.es/Estudiantes/BecasAyudasPremios/Becas/AyudasCAM",
    "https://www.upm.es/Estudiantes/BecasAyudasPremios/Becas/Becas_Movilidad",
    "https://www.upm.es/Estudiantes/BecasAyudasPremios/Becas/BecasColaboracionUPM",
    "https://www.upm.es/Estudiantes/BecasAyudasPremios/AyudasAlumnosDobleTitulacion",
    "https://www.upm.es/Estudiantes/Movilidad",
    "https://www.upm.es/Estudiantes/Movilidad/escuelaVeranoInt",
    "https://www.upm.es/Estudiantes/Movilidad/Programas_Nacionales",
    "https://www.upm.es/Estudiantes/Movilidad/Programas_Internacionales",
    "https://www.upm.es/Estudiantes/Movilidad/Coordinadores",
    "https://www.upm.es/Estudiantes/Practicas",
    "https://www.upm.es/Estudiantes/BecasAyudasPremios/AyudasRusia",
    "https://www.upm.es/Estudiantes/OrdenacionAcademica",
    "https://www.upm.es/Estudiantes/OrdenacionAcademica/Convenios",
    "https://www.upm.es/Estudiantes/Atencion/ManualesPrevencionRiesgos",
    "https://www.upm.es/Estudiantes/BecasAyudasPremios/AyudasConsejoSocial",
    "https://www.upm.es/Estudiantes/BecasAyudasPremios/AyudasViajes",
    "https://www.upm.es/UPM",
    "https://www.upm.es/Estudiantes/BecasAyudasPremios/Bolsa%20de%20viaje",
    "https://www.upm.es/Estudiantes/e-EdU/Puesta%20a%20Punto%20para%20Estudiantes",
    "https://www.upm.es/Estudiantes/Asociaciones/RelacionAsociaciones"
]
GSFS_BASE_URL = "https://www.upm.es/gsfs/"
ALLOWED_GSFS_URLS = [
    "https://www.upm.es/gsfs/SFS18800",
    "https://www.upm.es/gsfs/SFS34035",
    "https://www.upm.es/gsfs/SFS18435",
    "https://www.upm.es/gsfs/SFS04242",
    "https://www.upm.es/gsfs/SFS38713",
    "https://www.upm.es/gsfs/SFS38714",
    "https://www.upm.es/gsfs/SFS43815",
    "https://www.upm.es/gsfs/SFS38510",
    "https://www.upm.es/gsfs/SFS43083",
    "https://www.upm.es/gsfs/SFS44465"
]
ADDITIONAL_URLS = [
    "https://www.upm.es/sfs/Rectorado/Vicerrectorado%20de%20Alumnos/Secretaria/SeguroResponCivilAXA_Resumen.pdf",
    "https://www.upm.es/gsfs/SFS38713",
    "https://www.upm.es/gsfs/SFS38714",
    "https://www.upm.es/gsfs/SFS18435",
    "https://www.upm.es/gsfs/SFS34035",
    "https://ehea.info/page-full_members",
    "https://www.upm.es/sfs/Rectorado/Vicerrectorado%20de%20Alumnos/Extension%20Universitaria/Bolsa%20Vivienda/20230623-DOSSIER%20INFORMATIVO.pdf",
    "https://www.upm.es/UPM/ServiciosTecnologicos/Office365",
    "https://proyectos.crue.org/acreditacion/wp-content/uploads/sites/3/2026/01/tabla_equivalencias_ingles_19_01_2026.pdf",
    "https://www.upm.es/Estudiantes/Estudios_Titulaciones/Estudios_Master/?fmt=detail&id=CON24736",
    "https://www.etsist.upm.es/escuela/departamentos/LING/acreditacion-b2-en-lengua-inglesa",
    "https://www.upm.es/UPM/Accion_Social/AyudaTransporte",
    "https://www.upm.es/gsfs/SFS43815",
    "https://www.upm.es/gsfs/SFS38510",
    "https://www.upm.es/gsfs/SFS43083",
    "https://www.upm.es/gsfs/SFS44465",
    "https://www.upm.es/sfs/Rectorado/Vicerrectorado%20de%20Relaciones%20Internacionales/General/2017_Dobles_Titulaciones_Master.pdf",
    "https://www.upm.es/sfs/Rectorado/Vicerrectorado%20de%20Alumnos/Extension%20Universitaria/Intercambios:%20movilidad%20de%20estudiantes/Erasmus/EPP_2020-21_Coordinadores_en_los_Centros_v2.pdf",
    "https://www.fi.upm.es/?pagina=113",
    "https://www.fi.upm.es/?pagina=262",
    "https://www.fi.upm.es/?pagina=183",
    "https://www.fi.upm.es/?pagina=455",
    "https://www.upm.es/UPM/Biblioteca/NuestraBiblioteca/NormativaDocumentos?fmt=detail&prefmt=articulo&id=13160c5e1e23e210VgnVCM10000009c7648a____",
    "https://www.fi.upm.es/?id=proyectoinicio",
    "https://www.fi.upm.es/?id=comollegar"
]

JINA_FETCH_TIMEOUT = 30 # seconds
CRAWL_PRINT_ONLY = False
MODAL_TIMEOUT = 86400 # seconds

CHUNK_OVERLAP = 0 # if not decoder chunking (e.g., if SentenceSplitter)

CRAWL_MINUTES = 0
CRAWL_HOUR = 9
CRAWL_DAY = 10
CRAWL_MONTH = 9

REUSE_CRAWL = False
REUSE_CRAWL_PAST_CURRENT_YEAR = False
REUSE_TIMESTAMP = "20260203_161009" # reuse this within the same year if REUSE_CRAWL, forever if REUSE_CRAWL_PAST_CURRENT_YEAR
RECREATE_QDRANT_COLLECTIONS = True

FILE_START = "crawl_"
RAW_PATH = f"{VOLUME_PATH}/raw"
MANUALLY_CLEANED_PATH = f"{VOLUME_PATH}/manually_cleaned"
LM_CLEANED_PATH = f"{VOLUME_PATH}/lm_cleaned"

RAW_CHUNKS_PATH = f"{VOLUME_PATH}/raw_chunks"
MANUALLY_CLEANED_CHUNKS_PATH = f"{VOLUME_PATH}/manually_cleaned_chunks"
LM_CLEANED_TEXT_CHUNKS_PATH = f"{VOLUME_PATH}/lm_cleaned_text_chunks"
LM_ABSTRACT_CHUNKS_PATH = f"{VOLUME_PATH}/lm_abstract_chunks"
LM_SUMMARY_CHUNKS_PATH = f"{VOLUME_PATH}/lm_summary_chunks"
LM_Q_AND_A_CHUNKS_PATH = f"{VOLUME_PATH}/lm_q_and_a_chunks"

LM_CLEANED_TEXT_SUBCHUNKS_PATH = f"{VOLUME_PATH}/lm_cleaned_text_subchunks"
LM_SUMMARY_SUBCHUNKS_PATH = f"{VOLUME_PATH}/lm_summary_subchunks"
LM_Q_AND_A_VALID_CHUNKS_PATH = f"{VOLUME_PATH}/lm_q_and_a_valid_chunks"
LM_Q_AND_A_FOR_Q_ONLY_VALID_CHUNKS_PATH = f"{VOLUME_PATH}/lm_q_and_a_for_q_only_valid_chunks"

ENCODE_VARIANTS = {
    "raw_chunks": {
        "encoders": {
            "bm25": {"batch_size": 1024},
            "splade": {"batch_size": 1024},
            "colbert": {"batch_size": 1024},
            "bge_small": {"batch_size": 1024},
        },
    },
    "manually_cleaned_chunks": {
        "encoders": {
            "bm25": {"batch_size": 384},
            "splade": {"batch_size": 384},
            "colbert": {"batch_size": 384},
            "bge_small": {"batch_size": 384},
        },
    },
    "lm_cleaned_text_chunks": {
        "encoders": {
            "bm25": {"batch_size": 384},
            "splade": {"batch_size": 384},
            "colbert": {"batch_size": 384},
            "bge_small": {"batch_size": 384},
        },
    },
    "lm_summary_chunks": {
        "encoders": {
            "bm25": {"batch_size": 256},
            "splade": {"batch_size": 256},
            "colbert": {"batch_size": 256},
            "bge_small": {"batch_size": 256},
        },
    },
    "lm_q_and_a_chunks": {
        "encoders": {
            "bm25": {"batch_size": 64},
            "splade": {"batch_size": 64},
            "colbert": {"batch_size": 64},
            "bge_small": {"batch_size": 64},
        },
    },
    "lm_q_and_a_for_q_only_chunks": {
        "encoders": {
            "bm25": {"batch_size": 64},
            "splade": {"batch_size": 64},
            "colbert": {"batch_size": 64},
            "bge_small": {"batch_size": 64},
        },
    },
}

PYTHON_VERSION = "3.11"
PACKAGES = [
    "openai",
    "requests",
    "transformers",
    "torch",
    "llama-index-core"
]

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install(*PACKAGES)
    .add_local_python_source("config", "helpers")
)
