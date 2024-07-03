Search.setIndex({"docnames": ["examples", "examples/dumbbell_td_evp", "examples/filtered_power_iteration", "examples/filters", "installation", "intro"], "filenames": ["examples.md", "examples/dumbbell_td_evp.ipynb", "examples/filtered_power_iteration.ipynb", "examples/filters.ipynb", "installation.md", "intro.md"], "titles": ["<span class=\"section-number\">2. </span>Examples", "<span class=\"section-number\">2.3. </span>Resonances of a 2d dumbbell", "<span class=\"section-number\">2.2. </span>Filtered power iteration", "<span class=\"section-number\">2.1. </span>Time domain filters", "<span class=\"section-number\">1. </span>Installation", "Time domain methods for resonance problems"], "terms": {"The": [2, 3, 4, 5], "implement": [2, 3, 4, 5], "i": [1, 2, 3, 4, 5], "base": [1, 2, 3, 4, 5], "high": [4, 5], "order": [1, 2, 4, 5], "finit": [2, 4, 5], "element": [2, 4, 5], "librari": [4, 5], "ngsolv": [1, 2, 3, 4, 5], "which": [2, 5], "can": 5, "us": [4, 5], "python": 4, "m": [1, 2, 3, 5], "pip": [], "numpi": [1, 2, 3, 4], "scipi": 1, "matplotlib": [1, 2, 3, 4], "jupyt": 4, "ipyparallel": [], "scikit": [], "build": [], "upgrad": [], "webgui_jupyter_widget": [], "For": [4, 5], "troubleshoot": 4, "more": 5, "detail": 4, "we": [1, 2, 3, 4, 5], "refer": 4, "variou": 4, "tutori": 4, "ngs24": 4, "document": [1, 2, 3, 4], "To": [4, 5], "run": 4, "ipynb": 4, "exampl": [4, 5], "you": 4, "addition": 4, "need": 4, "an": [1, 2, 4, 5], "wess": [1, 2, 3, 5], "l": [1, 2, 3, 5], "nannen": [1, 2, 3, 5], "tu": 5, "wien": 5, "institut": 5, "analysi": 5, "scientif": 5, "comput": [2, 5], "thi": [1, 2, 3, 5], "book": 5, "design": [1, 2, 5], "provid": 5, "introduct": 5, "full": 5, "mathemat": 5, "nw24": [], "instal": 5, "lothar": [1, 2, 3, 5], "marku": [1, 2, 3, 5], "A": [1, 2, 3, 5], "krylov": [2, 3, 5], "solver": [1, 2, 3, 5], "filter": [0, 5], "solut": [1, 2, 3, 5], "2024": [1, 2, 3, 5], "arxiv": [1, 2, 3, 5], "2402": [1, 2, 3, 5], "08515": [1, 2, 3, 5], "exposit": 5, "extend": 5, "abstract": 5, "nw224": [], "short": 5, "read": 5, "preprint": [2, 3, 5], "comprehens": 5, "reson": [0, 2, 3], "wind": 5, "instrument": 5, "berlin": 5, "germani": 5, "edmond": 5, "mpdl": 5, "doi": 5, "10": [1, 2, 3, 5], "17617": 5, "3": [1, 2, 3, 5], "mbe4aa": 5, "1": [1, 2, 3], "2": [1, 2, 3], "laurent": 5, "gizon": 5, "editor": 5, "miss": [], "author": [], "nw24_wave": [], "In": [1, 5], "16th": 5, "intern": 5, "confer": 5, "numer": 5, "aspect": 5, "wave": 5, "propag": [2, 5], "298": 5, "299": 5, "pp": 5, "nw24b": 5, "nw24a": [1, 2, 3, 5], "eigenvalu": [1, 2, 3, 5], "asfd": [], "moreov": 4, "packag": 4, "ar": [2, 4], "solv": [1, 5], "problem": [1, 3], "harmon": 5, "type": 5, "equat": 5, "without": 5, "have": 5, "invert": 5, "larg": [2, 5], "system": 5, "matric": 5, "our": 5, "idea": [2, 5], "util": 5, "fast": 5, "explicit": 5, "oper": 5, "map": 5, "initi": [2, 5], "valu": [2, 5], "effici": 5, "auxiliari": [2, 5], "space": 5, "sought": 5, "after": 5, "origin": 5, "largest": [1, 5], "magnitud": 5, "while": 5, "eigenvector": 5, "two": [1, 5], "correspond": [2, 5], "stabil": 5, "follow": [2, 5], "from": [1, 2, 3, 5], "underli": 5, "algorithm": [1, 5], "time": [0, 2], "domain": [0, 1, 2], "notebook": [1, 2, 3], "part": [1, 2, 3], "td_evp": [1, 2, 3], "method": [1, 2, 3], "notat": [2, 3], "construct": 3, "discret": 3, "function": 3, "cf": [2, 3], "lemma": 3, "beta_": 3, "alpha": [1, 2, 3], "omega": [1, 2, 3], "tau": [1, 2, 3], "sum_": [2, 3], "0": [1, 2, 3], "q_l": 3, "given": [2, 3], "step": [2, 3], "weight": [1, 2, 3], "evolut": [2, 3], "14": [2, 3], "q_": 3, "quad": [2, 3], "q_0": 3, "dsaf": [], "import": [1, 2, 3], "np": [1, 2, 3], "pyplot": [1, 2, 3], "pl": [1, 2, 3], "def": [1, 2, 3], "beta": [1, 2, 3], "isscalar": [1, 2, 3], "q": [1, 2, 3], "els": [1, 2, 3], "ones": [1, 2, 3], "shape": [1, 2, 3], "q_old": [1, 2, 3], "out": [1, 2, 3], "q_new": [1, 2, 3], "return": [1, 2, 3], "pick": [2, 3], "goal": 3, "approxim": [2, 3], "characterist": 3, "w_min": [1, 2, 3], "5": [1, 2, 3], "w_max": [1, 2, 3], "02": [2, 3], "weightf": [1, 2, 3], "lambda": [1, 2, 3], "t": [1, 2, 3], "4": [1, 2, 3], "pi": [1, 2, 3], "co": [1, 2, 3], "sinc": [1, 2, 3], "20": [1, 3], "50": 3, "100": 3, "1000": 3, "arang": [1, 2, 3], "plot": [2, 3], "power": [0, 5], "iter": [0, 5], "present": [1, 2], "main": 2, "simpl": 2, "toi": 2, "dirichlet": 2, "laplacian": 2, "rectangl": [1, 2], "want": 2, "find": 2, "u": [1, 2], "delta": 2, "text": 2, "homogen": 2, "boundari": 2, "condit": 2, "eigenpair": 2, "omega_": 2, "k": 2, "sqrt": [1, 2], "2k": 2, "u_": 2, "sin": 2, "lx": 2, "2ky": 2, "ldot": 2, "galerkin": 2, "lead": 2, "gener": 2, "matrix": [1, 2], "mathbf": 2, "": 2, "c": [1, 2], "mu": [1, 2], "where": 2, "y_l": 2, "choos": 2, "mass": [1, 2], "lump": 2, "assembl": [1, 2], "stiff": [1, 2], "invers": [1, 2], "webgui": [1, 2], "draw": [1, 2], "netgen": [1, 2], "occ": [1, 2], "geo": [1, 2], "occgeometri": [1, 2], "face": [1, 2], "dim": [1, 2], "mesh": 2, "generatemesh": [1, 2], "maxh": [1, 2], "fe": [1, 2], "h1lumpingfespac": [1, 2], "default": 2, "print": [1, 2], "number": 2, "dof": 2, "ndof": [1, 2], "v": [1, 2], "tnt": [1, 2], "bilinearform": [1, 2], "dx": [1, 2], "intrul": [1, 2], "getintegrationrul": [1, 2], "mat": [1, 2], "grad": [1, 2], "massinv": [1, 2], "freedof": [1, 2], "889": 2, "some": 2, "paramet": 2, "take": 2, "look": 2, "result": 2, "note": 2, "real": [1, 2], "life": 2, "applic": 2, "chosen": 2, "possibl": 2, "still": 2, "obei": 2, "cfl": 2, "here": 2, "just": 2, "arbitrari": 2, "small": 2, "enough": 2, "thu": 2, "stabl": [1, 2], "8": 2, "300": 2, "01": [1, 2], "ev": 2, "arrai": 2, "rang": [1, 2], "o": 2, "xlim": [1, 2], "first": [1, 2], "continu": 2, "sort": [1, 2], "23606798": [], "82842712": [], "60555128": [], "12310563": [], "47213595": [], "38516481": [], "65685425": [], "6": [1, 2], "08276253": [], "abov": 2, "e": 2, "show": 2, "yield": 2, "eigenfunct": 2, "x": 2, "4y": 2, "17": 2, "One": 2, "data": [1, 2], "over": 2, "gridfunct": [1, 2], "vec": [1, 2], "setrandom": [1, 2], "tmpvec": [1, 2], "createvector": [1, 2], "u_old": 2, "u_new": 2, "scene": 2, "draweveri": 2, "uvec": 2, "len": [1, 2], "redraw": 2, "n": [1, 2], "norm": [1, 2], "normm": 2, "innerproduct": [1, 2], "approx_w": 2, "42997598669488": 2, "794993549187162": 2, "059381566808358": 2, "116294074589648": 2, "122283106734649": 2, "122844998915219": 2, "122897109099626": 2, "7": 2, "122901936525614": 2, "122902383683407": 2, "9": 2, "122902425102294": 2, "12290242893877": [], "11": [], "122902429294125": [], "12": [], "1229024293270395": [], "13": [], "12290242933009": [], "122902429330373": [], "15": 1, "122902429330399": [], "16": [], "122902429330401": [], "122902429330402": [], "18": [], "19": [], "drawscen": 2, "none": [1, 2], "2d": [0, 5], "dumbbel": [0, 5], "particular": 1, "dimension": 1, "section": 1, "bei": 1, "do": 1, "necessari": 1, "linalg": 1, "spl": 1, "ti": [], "special": [], "jn_zero": [], "jnjnp_zero": [], "jnp_zero": [], "class": 1, "filteredc": 1, "basematrix": 1, "__init__": 1, "self": 1, "mata": 1, "matm_inv": 1, "super": 1, "dt": 1, "nstep": 1, "vecu": 1, "createcolvector": 1, "tmpvec1": 1, "tmpvec2": 1, "col": 1, "mult": 1, "rh": 1, "taskmanag": 1, "unew": 1, "uold": 1, "r": 1, "format": 1, "end": 1, "r_l": 1, "r_r": 1, "d": 1, "03": 1, "w": 1, "circ_left": 1, "circl": 1, "circ_right": 1, "handl": 1, "moveto": 1, "curv": 1, "diagon": 1, "true": 1, "14613": 1, "tol": 1, "1e": 1, "tmpv": 1, "tmpv2": 1, "500": 1, "append": 1, "break": 1, "lammax": 1, "200": 1, "estim": 1, "49228": 1, "18216580044": 1, "009014115036549765": 1, "endt": 1, "maxstep": 1, "25": 1, "now": [], "errsmin": 1, "resmin": 1, "solveeveri": 1, "multivector": 1, "figur": 1, "krylowstep": 1, "appendorthogon": 1, "tvec": 1, "tvecs2": 1, "matm_proj": 1, "mats_proj": 1, "lam": 1, "eigh": 1, "ab": 1, "li": 1, "xr": 1, "01802823007309953": [], "027042345109649292": [], "03605646014619906": [], "045070575182748825": [], "05408469021929859": [], "06309880525584835": [], "07211292029239812": [], "08112703532894788": [], "09014115036549765": [], "09915526540204742": [], "10816938043859718": [], "11718349547514695": [], "1261976105116967": [], "13521172554824645": [], "1442258405847962": [], "15323995562134596": [], "1622540706578957": [], "17126818569444546": [], "18028230073099522": [], "18929641576754497": [], "21": [], "19831053080409472": [], "22": [], "20732464584064447": [], "23": [], "21633876087719422": [], "24": [], "22535287591374398": [], "23436699095029373": [], "26": [], "24338110598684348": [], "27": [], "25239522102339323": [], "28": [], "261409336059943": [], "29": [], "27042345109649274": [], "30": [], "2794375661330425": [], "31": [], "28845168116959224": [], "32": [], "297465796206142": [], "33": [], "30647991124269175": [], "34": [], "3154940262792415": [], "35": [], "32450814131579125": [], "36": [], "333522256352341": [], "37": [], "34253637138889076": [], "38": [], "3515504864254405": [], "39": [], "36056460146199026": [], "40": [], "36957871649854": [], "41": [], "37859283153508977": [], "42": [], "3876069465716395": [], "43": [], "3966210616081893": [], "44": [], "405635176644739": [], "45": [], "4146492916812888": [], "46": [], "42366340671783853": [], "47": [], "4326775217543883": [], "48": [], "44169163679093804": [], "49": [], "4507057518274878": [], "45971986686403754": [], "51": [], "4687339819005873": [], "52": [], "47774809693713705": [], "53": [], "4867622119736868": [], "54": [], "49577632701023655": [], "55": [], "5047904420467864": [], "56": [], "5138045570833362": [], "57": [], "522818672119886": [], "58": [], "5318327871564358": [], "59": [], "5408469021929856": [], "60": [], "5498610172295354": [], "61": [], "5588751322660852": [], "62": [], "567889247302635": [], "63": [], "5769033623391848": [], "64": [], "5859174773757346": [], "65": [], "5949315924122844": [], "66": [], "6039457074488342": [], "67": [], "612959822485384": [], "68": [], "6219739375219339": [], "69": [], "6309880525584837": [], "70": [], "6400021675950335": [], "71": [], "6490162826315833": [], "72": [], "6580303976681331": [], "73": [], "6670445127046829": [], "74": [], "6760586277412327": [], "75": [], "6850727427777825": [], "76": [], "6940868578143323": [], "77": [], "7031009728508821": [], "78": [], "7121150878874319": [], "79": [], "7211292029239817": [], "80": [], "7301433179605316": [], "81": [], "7391574329970814": [], "82": [], "7481715480336312": [], "83": [], "757185663070181": [], "84": [], "7661997781067308": [], "85": [], "7752138931432806": [], "86": [], "7842280081798304": [], "87": [], "7932421232163802": [], "88": [], "80225623825293": [], "89": [], "8112703532894798": [], "90": [], "8202844683260296": [], "91": [], "8292985833625794": [], "92": [], "8383126983991293": [], "93": [], "8473268134356791": [], "94": [], "8563409284722289": [], "95": [], "8653550435087787": [], "96": [], "8743691585453285": [], "97": [], "8833832735818783": [], "98": [], "8923973886184281": [], "99": [], "9014115036549779": [], "9104256186915277": [], "101": [], "9194397337280775": [], "102": [], "9284538487646273": [], "103": [], "9374679638011771": [], "104": [], "946482078837727": [], "105": [], "9554961938742768": [], "106": [], "9645103089108266": [], "107": [], "9735244239473764": [], "108": [], "9825385389839262": [], "109": [], "991552654020476": [], "110": [], "0005667690570257": [], "111": [], "0095808840935754": [], "112": [], "018594999130125": [], "113": [], "0276091141666748": [], "114": [], "0366232292032245": [], "115": [], "0456373442397742": [], "116": [], "0546514592763239": [], "117": [], "0636655743128736": [], "118": [], "0726796893494233": [], "119": [], "081693804385973": [], "120": [], "0907079194225227": [], "121": [], "0997220344590724": [], "122": [], "108736149495622": [], "123": [], "1177502645321717": [], "124": [], "1267643795687214": [], "125": [], "1357784946052711": [], "126": [], "1447926096418208": [], "127": [], "1538067246783705": [], "128": [], "1628208397149202": [], "129": [], "17183495475147": [], "130": [], "1808490697880196": [], "131": [], "1898631848245693": [], "132": [], "198877299861119": [], "133": [], "2078914148976687": [], "134": [], "2169055299342184": [], "135": [], "225919644970768": [], "136": [], "2349337600073178": [], "137": [], "2439478750438675": [], "138": [], "2529619900804172": [], "139": [], "261976105116967": [], "140": [], "2709902201535166": [], "141": [], "2800043351900663": [], "142": [], "289018450226616": [], "143": [], "2980325652631657": [], "144": [], "3070466802997154": [], "145": [], "316060795336265": [], "146": [], "3250749103728148": [], "147": [], "3340890254093645": [], "148": [], "3431031404459142": [], "149": [], "3521172554824639": [], "150": [], "3611313705190136": [], "151": [], "3701454855555633": [], "152": [], "379159600592113": [], "153": [], "3881737156286627": [], "154": [], "3971878306652123": [], "155": [], "406201945701762": [], "156": [], "4152160607383117": [], "157": [], "4242301757748614": [], "158": [], "4332442908114111": [], "159": [], "4422584058479608": [], "160": [], "4512725208845105": [], "161": [], "4602866359210602": [], "162": [], "46930075095761": [], "163": [], "4783148659941596": [], "164": [], "4873289810307093": [], "165": [], "496343096067259": [], "166": [], "5053572111038087": [], "167": [], "5143713261403584": [], "168": [], "523385441176908": [], "169": [], "5323995562134578": [], "170": [], "5414136712500075": [], "171": [], "5504277862865572": [], "172": [], "559441901323107": [], "173": [], "5684560163596566": [], "174": [], "5774701313962063": [], "175": [], "586484246432756": [], "176": [], "5954983614693057": [], "177": [], "6045124765058554": [], "178": [], "613526591542405": [], "179": [], "6225407065789548": [], "180": [], "6315548216155045": [], "181": [], "6405689366520542": [], "182": [], "6495830516886039": [], "183": [], "6585971667251536": [], "184": [], "6676112817617033": [], "185": [], "676625396798253": [], "186": [], "6856395118348027": [], "187": [], "6946536268713523": [], "188": [], "703667741907902": [], "189": [], "7126818569444517": [], "190": [], "7216959719810014": [], "191": [], "7307100870175511": [], "192": [], "7397242020541008": [], "193": [], "7487383170906505": [], "194": [], "7577524321272002": [], "195": [], "76676654716375": [], "196": [], "7757806622002996": [], "197": [], "7847947772368493": [], "198": [], "793808892273399": [], "199": [], "eigf": 1, "evalu": 1, "gfu": 1, "7068135620766916e": 1, "06": 1, "2015955703259844": 1, "2275073463471204": 1, "9051215442919904": 1, "0363109063162037": 1, "2840876468299025": 1, "579927270284838": 1, "801083497774604": 1, "889397348173861": 1, "5454948557875094": 1, "re": 2, "23606797749979": 2, "8284271247461903": 2, "605551275463989": 2, "123105625617661": 2, "47213595499958": 2, "385164807134504": 2, "656854249492381": 2, "082762530298219": 2}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"exampl": 0, "instal": 4, "time": [1, 3, 5], "domain": [3, 5], "method": 5, "eigenvalu": [], "problem": [2, 5], "tabl": 5, "content": 5, "refer": 5, "comput": [], "reson": [1, 5], "abstract": [], "filter": [1, 2, 3], "m": [], "wess": [], "2024": [], "power": [1, 2], "iter": [1, 2], "introduct": [1, 2], "set": 2, "definit": 2, "defin": [1, 2], "space": 2, "bilinear": 2, "form": 2, "discret": 2, "function": [1, 2], "oper": [1, 2], "appli": 2, "2d": 1, "dumbbel": 1, "implement": 1, "The": 1, "geometri": 1, "mesh": 1, "determin": 1, "step": 1, "plot": 1, "It": 1, "remain": 1, "start": 1, "krylov": 1, "loop": 1, "approxim": 1, "eigenfunct": 1}, "envversion": {"sphinx.domains.c": 3, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 9, "sphinx.domains.index": 1, "sphinx.domains.javascript": 3, "sphinx.domains.math": 2, "sphinx.domains.python": 4, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinxcontrib.bibtex": 9, "sphinx": 60}, "alltitles": {"Examples": [[0, "examples"]], "Resonances of a 2d dumbbell": [[1, "resonances-of-a-2d-dumbbell"]], "Introduction": [[1, "introduction"], [2, "introduction"]], "Implementation": [[1, "implementation"]], "The filtered operator": [[1, "the-filtered-operator"]], "Geometry and mesh": [[1, "geometry-and-mesh"]], "Determine time step by power iteration": [[1, "determine-time-step-by-power-iteration"]], "Define and plot filter function": [[1, "define-and-plot-filter-function"]], "It remains to start the Krylov loop": [[1, "it-remains-to-start-the-krylov-loop"]], "Plotting the approximated eigenfunctions": [[1, "plotting-the-approximated-eigenfunctions"]], "Filtered power iteration": [[2, "filtered-power-iteration"]], "Problem setting and definitions": [[2, "problem-setting-and-definitions"]], "Defining spaces and bilinear forms": [[2, "defining-spaces-and-bilinear-forms"]], "Discrete filter function": [[2, "discrete-filter-function"]], "Defining the filtered operator": [[2, "defining-the-filtered-operator"]], "Applying the power iteration": [[2, "applying-the-power-iteration"]], "Time domain filters": [[3, "time-domain-filters"]], "Installation": [[4, "installation"]], "Time domain methods for resonance problems": [[5, "time-domain-methods-for-resonance-problems"]], "Table of Contents": [[5, "table-of-contents"]], "References": [[5, "references"]]}, "indexentries": {}})