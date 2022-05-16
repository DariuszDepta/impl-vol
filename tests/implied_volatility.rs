use impl_vol::*;

fn eq(_expected: f64, _actual: f64) {
  //assert!((expected - actual).abs() < f64::EPSILON, "expected: {}\n  actual: {},", expected, actual);
}

#[test]
#[rustfmt::skip]
fn test_iv_implied_volatility_from_a_transformed_rational_guess() {
  let f = iv_implied_volatility_from_a_transformed_rational_guess;
  eq(0.3148253556850184, f(539.1269453050334, 2170.4221251767294, 1700.00, 0.926027, 1.0));
  eq(0.3005835339311901, f(459.18797785046036, 2170.4221251767294, 1800.00, 0.926027, 1.0));
  eq(0.28700565680447726, f(383.9915991044646, 2170.4221251767294, 1900.00, 0.926027, 1.0));
  eq(0.2743335201142195, f(314.5390222388568, 2170.4221251767294, 2000.00, 0.926027, 1.0));
  eq(0.25169991491284416, f(242.92593273934156, 2170.4221251767294, 2100.00, 0.926027, 1.0));
  eq(0.23327460982662723, f(180.85071608707597, 2170.4221251767294, 2200.00, 0.926027, 1.0));
  eq(0.2250561112015883, f(134.79491018378215, 2170.4221251767294, 2300.00, 0.926027, 1.0));
  eq(0.21984446432090532, f(98.9620177189769, 2170.4221251767294, 2400.00, 0.926027, 1.0));
  eq(0.21738114452619214, f(72.29813009075414, 2170.4221251767294, 2500.00, 0.926027, 1.0));
  eq(3.156649599921243, f(655.780313742833, 2104.57774868891, 1450.00, 0.00274, 1.0));
  eq(3.02617150392871, f(630.7772516053145, 2104.57774868891, 1475.00, 0.00274, 1.0));
  eq(2.8975609692697217, f(605.7741894677962, 2104.57774868891, 1500.00, 0.00274, 1.0));
  eq(2.7707295863322567, f(580.7711273302777, 2104.57774868891, 1525.00, 0.00274, 1.0));
  eq(2.6455909325185045, f(555.7680651927593, 2104.57774868891, 1550.00, 0.00274, 1.0));
  eq(2.52206008175039, f(530.7650030552409, 2104.57774868891, 1575.00, 0.00274, 1.0));
  eq(2.400053080604957, f(505.7619409177224, 2104.57774868891, 1600.00, 0.00274, 1.0));
  eq(2.2794863752310346, f(480.75887878020404, 2104.57774868891, 1625.00, 0.00274, 1.0));
  eq(2.1602761686971643, f(455.7558166426856, 2104.57774868891, 1650.00, 0.00274, 1.0));
  eq(2.042337681975436, f(430.7527545051672, 2104.57774868891, 1675.00, 0.00274, 1.0));
  eq(1.6304380276058712, f(404.8995882549732, 2104.57774868891, 1700.00, 0.00274, 1.0));
  eq(1.80992643000417, f(380.7466302301304, 2104.57774868891, 1725.00, 0.00274, 1.0));
  eq(1.2146964770768587, f(354.6434333585612, 2104.57774868891, 1750.00, 0.00274, 1.0));
  eq(1.5815164754122906, f(330.74050595509357, 2104.57774868891, 1775.00, 0.00274, 1.0));
  eq(1.0342741992659326, f(304.63730908352437, 2104.57774868891, 1800.00, 0.00274, 1.0));
  eq(1.3562735519417453, f(280.73438168005674, 2104.57774868891, 1825.00, 0.00274, 1.0));
  eq(1.244532348725751, f(255.73131954253833, 2104.57774868891, 1850.00, 0.00274, 1.0));
  eq(1.1331792595442773, f(230.72825740501992, 2104.57774868891, 1875.00, 0.00274, 1.0));
  eq(0.8204988703734793, f(204.82508503055084, 2104.57774868891, 1900.00, 0.00274, 1.0));
  eq(0.7261575194492099, f(179.82202289303243, 2104.57774868891, 1925.00, 0.00274, 1.0));
  eq(0.6320297064892296, f(154.818960755514, 2104.57774868891, 1950.00, 0.00274, 1.0));
  eq(0.5958680865558811, f(130.0659292393708, 2104.57774868891, 1975.00, 0.00274, 1.0));
  eq(0.5294440140729184, f(105.31289772322755, 2104.57774868891, 2000.00, 0.00274, 1.0));
  eq(0.496388440997469, f(81.2099458226598, 2104.57774868891, 2025.00, 0.00274, 1.0));
  eq(0.43568760638112863, f(57.50704291629235, 2104.57774868891, 2050.00, 0.00274, 1.0));
  eq(0.3833302559436945, f(35.51434946013115, 2104.57774868891, 2075.00, 0.00274, 1.0));
  eq(0.321294483187517, f(16.512022235617163, 2104.57774868891, 2100.00, 0.00274, 1.0));
  eq(0.30448152962469793, f(5.630689593369146, 2104.57774868891, 2125.00, 0.00274, 1.0));
  eq(0.3214700368943528, f(1.7402131247712815, 2104.57774868891, 2150.00, 0.00274, 1.0));
  eq(0.4302029210097354, f(1.5401886276711343, 2104.57774868891, 2175.00, 0.00274, 1.0));
  eq(0.445169852050396, f(0.5500673670254052, 2104.57774868891, 2200.00, 0.00274, 1.0));
  eq(0.493259600805132, f(0.3100379705052283, 2104.57774868891, 2225.00, 0.00274, 1.0));
  eq(0.5605833299002175, f(0.2500306213751841, 2104.57774868891, 2250.00, 0.00274, 1.0));
  eq(0.6553373226155671, f(0.3000367456502209, 2104.57774868891, 2275.00, 0.00274, 1.0));
  eq(0.6644757991705206, f(0.13001592311509574, 2104.57774868891, 2300.00, 0.00274, 1.0));
  eq(0.7628183869400739, f(0.18002204739013256, 2104.57774868891, 2325.00, 0.00274, 1.0));
  eq(0.8291564777805057, f(0.1700208225351252, 2104.57774868891, 2350.00, 0.00274, 1.0));
  eq(0.9047019742509538, f(0.18002204739013256, 2104.57774868891, 2375.00, 0.00274, 1.0));
  eq(0.9735706275145919, f(0.18002204739013256, 2104.57774868891, 2400.00, 0.00274, 1.0));
  eq(1.0411802623686313, f(0.18002204739013256, 2104.57774868891, 2425.00, 0.00274, 1.0));
  eq(1.1009177441264046, f(0.1700208225351252, 2104.57774868891, 2450.00, 0.00274, 1.0));
  eq(1.1659088594872704, f(0.1700208225351252, 2104.57774868891, 2475.00, 0.00274, 1.0));
  eq(1.3601070453836441, f(0.4200514439103093, 2104.57774868891, 2500.00, 0.00274, 1.0));
  eq(1.6657676343912893, f(1.340164130570987, 2104.57774868891, 2525.00, 0.00274, 1.0));
  eq(1.741060747812004, f(1.340164130570987, 2104.57774868891, 2550.00, 0.00274, 1.0));
  eq(1.8131434885275455, f(1.3301629057159796, 2104.57774868891, 2575.00, 0.00274, 1.0));
  eq(1.88594812556506, f(1.3301629057159796, 2104.57774868891, 2600.00, 0.00274, 1.0));
  eq(1.9576179947589516, f(1.3301629057159796, 2104.57774868891, 2625.00, 0.00274, 1.0));
  eq(2.0281995863237654, f(1.3301629057159796, 2104.57774868891, 2650.00, 0.00274, 1.0));
  eq(0.5178883470574448, f(665.462984359371, 2114.3697225533924, 1450.00, 0.10137, 1.0));
  eq(0.48507298658100845, f(615.6261067748554, 2114.3697225533924, 1500.00, 0.10137, 1.0));
  eq(0.4443923741330698, f(565.6385128266768, 2114.3697225533924, 1550.00, 0.10137, 1.0));
  eq(0.4228318355491343, f(516.1030679694866, 2114.3697225533924, 1600.00, 0.10137, 1.0));
  eq(0.4021809805104577, f(466.768578263847, 2114.3697225533924, 1650.00, 0.10137, 1.0));
  eq(0.38353835632012456, f(417.7857600734207, 2114.3697225533924, 1700.00, 0.10137, 1.0));
  eq(0.3676721503024618, f(369.40580733764574, 2114.3697225533924, 1750.00, 0.10137, 1.0));
  eq(0.35893227834267866, f(322.38230187483657, 2114.3697225533924, 1800.00, 0.10137, 1.0));
  eq(0.3438379137800945, f(276.01190065456626, 2114.3697225533924, 1850.00, 0.10137, 1.0));
  eq(0.328172838629933, f(231.0481854951493, 2114.3697225533924, 1900.00, 0.10137, 1.0));
  eq(0.3139830536774275, f(188.3452157906751, 2114.3697225533924, 1950.00, 0.10137, 1.0));
  eq(0.30162767507578825, f(148.80728972312073, 2114.3697225533924, 2000.00, 0.10137, 1.0));
  eq(0.28547950595440386, f(111.98225820149769, 2114.3697225533924, 2050.00, 0.10137, 1.0));
  eq(0.2544430486906813, f(75.50889819508794, 2114.3697225533924, 2100.00, 0.10137, 1.0));
  eq(0.22830157966228282, f(45.626867159533504, 2114.3697225533924, 2150.00, 0.10137, 1.0));
  eq(0.22237047363881032, f(27.430378186638734, 2114.3697225533924, 2200.00, 0.10137, 1.0));
  eq(0.21738106367592602, f(15.202257214792821, 2114.3697225533924, 2250.00, 0.10137, 1.0));
  eq(0.21399792731678968, f(7.837250910468209, 2114.3697225533924, 2300.00, 0.10137, 1.0));
  eq(0.21074388620694393, f(3.6875270309510677, 2114.3697225533924, 2350.00, 0.10137, 1.0));
  eq(0.20746196828632452, f(1.567450182093642, 2114.3697225533924, 2400.00, 0.10137, 1.0));
  eq(0.2296484395247883, f(1.3564472729656516, 2114.3697225533924, 2450.00, 0.10137, 1.0));
  eq(0.2553625408850053, f(1.3463995153881283, 2114.3697225533924, 2500.00, 0.10137, 1.0));
  eq(0.27183955865275944, f(1.0851578183725212, 2114.3697225533924, 2550.00, 0.10137, 1.0));
  eq(0.44268574506046493, f(626.1840118916077, 2122.082345327525, 1500.00, 0.178082, 1.0));
  eq(0.4240944652699312, f(577.2476203725246, 2122.082345327525, 1550.00, 0.178082, 1.0));
  eq(0.40608429760795844, f(528.6139281824255, 2122.082345327525, 1600.00, 0.178082, 1.0));
  eq(0.3907829451933676, f(480.5856346502945, 2122.082345327525, 1650.00, 0.178082, 1.0));
  eq(0.3788994507921304, f(433.5158889932795, 2122.082345327525, 1700.00, 0.178082, 1.0));
  eq(0.36591015628252743, f(387.15244177056053, 2122.082345327525, 1750.00, 0.178082, 1.0));
  eq(0.35085090681066916, f(341.44484309397365, 2122.082345327525, 1800.00, 0.178082, 1.0));
  eq(0.33270429498506854, f(296.24174329902684, 2122.082345327525, 1850.00, 0.178082, 1.0));
  eq(0.3169882479054649, f(252.85483947798403, 2122.082345327525, 1900.00, 0.178082, 1.0));
  eq(0.30353367806184156, f(211.83908040064944, 2122.082345327525, 1950.00, 0.178082, 1.0));
  eq(0.2911407289926108, f(173.49716539600698, 2122.082345327525, 2000.00, 0.178082, 1.0));
  eq(0.27804494276044445, f(137.82909446405674, 2122.082345327525, 2050.00, 0.178082, 1.0));
  eq(0.25431008489820833, f(101.80787431495847, 2122.082345327525, 2100.00, 0.178082, 1.0));
  eq(0.23496845214754494, f(71.23524208757252, 2122.082345327525, 2150.00, 0.178082, 1.0));
  eq(0.229051038425291, f(50.06646901395677, 2122.082345327525, 2200.00, 0.178082, 1.0));
  eq(0.22402394929677796, f(33.75097518171815, 2122.082345327525, 2250.00, 0.178082, 1.0));
  eq(0.21937676896877012, f(21.693451910521375, 2122.082345327525, 2300.00, 0.178082, 1.0));
  eq(0.21506050472633984, f(13.24814063186724, 2122.082345327525, 2350.00, 0.178082, 1.0));
  eq(0.21353610949467353, f(8.031622195709309, 2122.082345327525, 2400.00, 0.178082, 1.0));
  eq(0.20959542692570957, f(4.41941020316668, 2122.082345327525, 2450.00, 0.178082, 1.0));
  eq(0.21005987348164243, f(2.572944296364163, 2122.082345327525, 2500.00, 0.178082, 1.0));
  eq(0.2097145055300775, f(1.42268684622489, 2122.082345327525, 2550.00, 0.178082, 1.0));
  eq(0.21850692370199207, f(1.0493576738112667, 2122.082345327525, 2600.00, 0.178082, 1.0));
  eq(0.21696781825850886, f(0.5448587921712346, 2122.082345327525, 2650.00, 0.178082, 1.0));
  eq(0.26088509964846995, f(1.3318770475296846, 2122.082345327525, 2700.00, 0.178082, 1.0));
  eq(0.24805477644336507, f(0.5448587921712346, 2122.082345327525, 2750.00, 0.178082, 1.0));
  eq(0.25290733225991063, f(0.3834191500464243, 2122.082345327525, 2800.00, 0.178082, 1.0));
  eq(0.41124517537850763, f(654.7666649294243, 2133.480231762442, 1500.00, 0.427397, 1.0));
  eq(0.3860926174899782, f(562.8740798200954, 2133.480231762442, 1600.00, 0.427397, 1.0));
  eq(0.36141987629703914, f(473.64356402869697, 2133.480231762442, 1700.00, 0.427397, 1.0));
  eq(0.3392923061131221, f(388.71331405857063, 2133.480231762442, 1800.00, 0.427397, 1.0));
  eq(0.3197836689359561, f(309.670332772329, 2133.480231762442, 1900.00, 0.427397, 1.0));
  eq(0.3035914171588155, f(238.664753080608, 2133.480231762442, 2000.00, 0.427397, 1.0));
  eq(0.25695887601470463, f(159.05864174634257, 2133.480231762442, 2100.00, 0.427397, 1.0));
  eq(0.21965579073664832, f(93.58197525340015, 2133.480231762442, 2200.00, 0.427397, 1.0));
  eq(0.215643014415276, f(58.565524994469236, 2133.480231762442, 2300.00, 0.427397, 1.0));
  eq(0.212456555145817, f(34.66833350197077, 2133.480231762442, 2400.00, 0.427397, 1.0));
  eq(0.20950754282579842, f(19.320480011287316, 2133.480231762442, 2500.00, 0.427397, 1.0));
  eq(0.20757247923192873, f(10.300160514761547, 2133.480231762442, 2600.00, 0.427397, 1.0));
  eq(0.20191156086956102, f(4.699576218961779, 2133.480231762442, 2700.00, 0.427397, 1.0));
  eq(0.19966394865701478, f(2.160371638781994, 2133.480231762442, 2800.00, 0.427397, 1.0));
  eq(0.3747403424374556, f(684.4413734370137, 2151.695636331533, 1500.00, 0.676712, 1.0));
  eq(0.35413761458177745, f(595.0038225902587, 2151.695636331533, 1600.00, 0.676712, 1.0));
  eq(0.33519089141991903, f(508.8903154451252, 2151.695636331533, 1700.00, 0.676712, 1.0));
  eq(0.31771442812234557, f(426.9838011098566, 2151.695636331533, 1800.00, 0.676712, 1.0));
  eq(0.301615737097542, f(350.3230432412094, 2151.695636331533, 1900.00, 0.676712, 1.0));
  eq(0.28741909753733086, f(280.41424914148104, 2151.695636331533, 2000.00, 0.676712, 1.0));
  eq(0.2507377324276666, f(201.67596395932043, 2151.695636331533, 2100.00, 0.676712, 1.0));
  eq(0.2277147044229386, f(139.3501445539162, 2151.695636331533, 2200.00, 0.676712, 1.0));
  eq(0.22200097660645288, f(98.42285647770079, 2151.695636331533, 2300.00, 0.676712, 1.0));
  eq(0.21737840351055399, f(67.20800859216087, 2151.695636331533, 2400.00, 0.676712, 1.0));
  eq(0.21368581226954708, f(44.44869687262076, 2151.695636331533, 2500.00, 0.676712, 1.0));
  eq(0.21047824290728326, f(28.420573648864313, 2151.695636331533, 2600.00, 0.676712, 1.0));
  eq(0.20830918074494287, f(17.794021440242897, 2151.695636331533, 2700.00, 0.676712, 1.0));
  eq(0.20498048988423265, f(10.491512933243039, 2151.695636331533, 2800.00, 0.676712, 1.0));
  eq(0.3149444210823517, f(68.76753627436892, 2170.4221251767294, 1700.00, 0.926027, -1.0));
  eq(0.300712632262775, f(88.84449514067893, 2170.4221251767294, 1800.00, 0.926027, -1.0));
  eq(0.2870665551751435, f(113.6113472854708, 2170.4221251767294, 1900.00, 0.926027, -1.0));
  eq(0.2744099970977063, f(144.1746967407459, 2170.4221251767294, 2000.00, 0.926027, -1.0));
  eq(0.2517260040130328, f(172.52483813201826, 2170.4221251767294, 2100.00, 0.926027, -1.0));
  eq(0.23331902338456798, f(210.4655478006356, 2170.4221251767294, 2200.00, 0.926027, -1.0));
  eq(0.22512039031279982, f(264.4256682182247, 2170.4221251767294, 2300.00, 0.926027, -1.0));
  eq(0.21986516962949168, f(328.55600664420706, 2170.4221251767294, 2400.00, 0.926027, -1.0));
  eq(0.21742640958863418, f(401.9080453368673, 2170.4221251767294, 2500.00, 0.926027, -1.0));
  eq(3.159293472938379, f(1.2101482074558911, 2104.57774868891, 1450.00, 0.00274, -1.0));
  eq(3.02975981903499, f(1.2101482074558911, 2104.57774868891, 1475.00, 0.00274, -1.0));
  eq(2.8987711314122224, f(1.2001469826008837, 2104.57774868891, 1500.00, 0.00274, -1.0));
  eq(2.775999245515769, f(1.2101482074558911, 2104.57774868891, 1525.00, 0.00274, -1.0));
  eq(2.6485773507179187, f(1.2001469826008837, 2104.57774868891, 1550.00, 0.00274, -1.0));
  eq(2.5258316962488405, f(1.2001469826008837, 2104.57774868891, 1575.00, 0.00274, -1.0));
  eq(2.4045419796248124, f(1.2001469826008837, 2104.57774868891, 1600.00, 0.00274, -1.0));
  eq(2.284625006507188, f(1.2001469826008837, 2104.57774868891, 1625.00, 0.00274, -1.0));
  eq(2.163414364647595, f(1.1901457577458765, 2104.57774868891, 1650.00, 0.00274, -1.0));
  eq(2.0461033927027947, f(1.1901457577458765, 2104.57774868891, 1675.00, 0.00274, -1.0));
  eq(1.6351422583993305, f(0.33004042021524305, 2104.57774868891, 1700.00, 0.00274, -1.0));
  eq(1.812485001060158, f(1.180144532890869, 2104.57774868891, 1725.00, 0.00274, -1.0));
  eq(1.185509520968369, f(0.050006124275036826, 2104.57774868891, 1750.00, 0.00274, -1.0));
  eq(1.5830336252213377, f(1.1701433080358616, 2104.57774868891, 1775.00, 0.00274, -1.0));
  eq(1.063255464377132, f(0.08000979884005892, 2104.57774868891, 1800.00, 0.00274, -1.0));
  eq(1.356916747599217, f(1.1601420831808542, 2104.57774868891, 1825.00, 0.00274, -1.0));
  eq(1.2439414504959638, f(1.150140858325847, 2104.57774868891, 1850.00, 0.00274, -1.0));
  eq(1.1331201772765422, f(1.150140858325847, 2104.57774868891, 1875.00, 0.00274, -1.0));
  eq(0.8255854579242277, f(0.2600318462301915, 2104.57774868891, 1900.00, 0.00274, -1.0));
  eq(0.7319131172110781, f(0.2600318462301915, 2104.57774868891, 1925.00, 0.00274, -1.0));
  eq(0.6382183551596223, f(0.2600318462301915, 2104.57774868891, 1950.00, 0.00274, -1.0));
  eq(0.5924350524469787, f(0.47005756818534616, 2104.57774868891, 1975.00, 0.00274, -1.0));
  eq(0.5300686055788173, f(0.740090639270545, 2104.57774868891, 2000.00, 0.00274, -1.0));
  eq(0.4955504144755588, f(1.6201984265111933, 2104.57774868891, 2025.00, 0.00274, -1.0));
  eq(0.4366294905997267, f(2.950361332227173, 2104.57774868891, 2050.00, 0.00274, -1.0));
  eq(0.38345152965308266, f(5.9407275638743755, 2104.57774868891, 2075.00, 0.00274, -1.0));
  eq(0.32122988214060244, f(11.931461252023785, 2104.57774868891, 2100.00, 0.00274, -1.0));
  eq(0.30448832714635776, f(26.053190747294188, 2104.57774868891, 2125.00, 0.00274, -1.0));
  eq(0.3216367578231006, f(47.16577641621473, 2104.57774868891, 2150.00, 0.00274, -1.0));
  eq(0.4299665284937345, f(71.95881283177799, 2104.57774868891, 2175.00, 0.00274, -1.0));
  eq(0.44372362255200226, f(95.96175248379568, 2104.57774868891, 2200.00, 0.00274, -1.0));
  eq(0.48923330247156727, f(120.7147839999389, 2104.57774868891, 2225.00, 0.00274, -1.0));
  eq(0.5592691967899464, f(145.66784001318229, 2104.57774868891, 2250.00, 0.00274, -1.0));
  eq(0.6549396586645222, f(170.7209082749757, 2104.57774868891, 2275.00, 0.00274, -1.0));
  eq(0.67608137752176, f(195.57395203966902, 2104.57774868891, 2300.00, 0.00274, -1.0));
  eq(0.7743014377986397, f(220.62702030146247, 2104.57774868891, 2325.00, 0.00274, -1.0));
  eq(0.8223421389268332, f(245.58007631470588, 2104.57774868891, 2350.00, 0.00274, -1.0));
  eq(0.8936083399226362, f(270.58313845222426, 2104.57774868891, 2375.00, 0.00274, -1.0));
  eq(0.9637531994295445, f(295.5862005897427, 2104.57774868891, 2400.00, 0.00274, -1.0));
  eq(1.0328481776242806, f(320.5892627272611, 2104.57774868891, 2425.00, 0.00274, -1.0));
  eq(1.10095375422797, f(345.5923248647795, 2104.57774868891, 2450.00, 0.00274, -1.0));
  eq(1.1681219494235855, f(370.5953870022979, 2104.57774868891, 2475.00, 0.00274, -1.0));
  eq(1.362497504412788, f(395.8484797611915, 2104.57774868891, 2500.00, 0.00274, -1.0));
  eq(1.6637798871354958, f(421.7516521356606, 2104.57774868891, 2525.00, 0.00274, -1.0));
  eq(1.739592675792554, f(446.754714273179, 2104.57774868891, 2550.00, 0.00274, -1.0));
  eq(1.814199273573531, f(471.75777641069743, 2104.57774868891, 2575.00, 0.00274, -1.0));
  eq(1.887654353386955, f(496.7608385482158, 2104.57774868891, 2600.00, 0.00274, -1.0));
  eq(1.9600075029418214, f(521.7639006857343, 2104.57774868891, 2625.00, 0.00274, -1.0));
  eq(2.031303962825265, f(546.7669628232527, 2104.57774868891, 2650.00, 0.00274, -1.0));
  eq(0.519850331232684, f(1.1253488486826146, 2114.3697225533924, 1450.00, 0.10137, -1.0));
  eq(0.48557689130952575, f(1.2660174547679415, 2114.3697225533924, 1500.00, 0.10137, -1.0));
  eq(0.4442579484011604, f(1.2660174547679415, 2114.3697225533924, 1550.00, 0.10137, -1.0));
  eq(0.4233780036440663, f(1.748309818489062, 2114.3697225533924, 1600.00, 0.10137, -1.0));
  eq(0.4016967665873546, f(2.381318545873033, 2114.3697225533924, 1650.00, 0.10137, -1.0));
  eq(0.3841591464566587, f(3.4463808490905077, 2114.3697225533924, 1700.00, 0.10137, -1.0));
  eq(0.3679431655250154, f(5.054022061494242, 2114.3697225533924, 1750.00, 0.10137, -1.0));
  eq(0.3592114111278033, f(8.038206062018675, 2114.3697225533924, 1800.00, 0.10137, -1.0));
  eq(0.3440331056201355, f(11.665446547504603, 2114.3697225533924, 1850.00, 0.10137, -1.0));
  eq(0.3283116202289175, f(16.6993730938438, 2114.3697225533924, 1900.00, 0.10137, -1.0));
  eq(0.3140288661528149, f(23.983997337548224, 2114.3697225533924, 1950.00, 0.10137, -1.0));
  eq(0.30174660726420605, f(34.46380849090507, 2114.3697225533924, 2000.00, 0.10137, -1.0));
  eq(0.2854140955612732, f(47.59622764472808, 2114.3697225533924, 2050.00, 0.10137, -1.0));
  eq(0.2546369366703804, f(61.19084364711717, 2114.3697225533924, 2100.00, 0.10137, -1.0));
  eq(0.2284124149498455, f(81.28635880216386, 2114.3697225533924, 2150.00, 0.10137, -1.0));
  eq(0.22248525960318433, f(113.08751153502524, 2114.3697225533924, 2200.00, 0.10137, -1.0));
  eq(0.21729619826323698, f(150.81684123862541, 2114.3697225533924, 2250.00, 0.10137, -1.0));
  eq(0.2140135803491801, f(193.469572155212, 2114.3697225533924, 2300.00, 0.10137, -1.0));
  eq(0.21098506392413965, f(239.33758549660607, 2114.3697225533924, 2350.00, 0.10137, -1.0));
  eq(0.20784428698373994, f(287.21515035350484, 2114.3697225533924, 2400.00, 0.10137, -1.0));
  eq(0.23004413491363995, f(337.001789150133, 2114.3697225533924, 2450.00, 0.10137, -1.0));
  eq(0.25572532663771547, f(386.98938309831163, 2114.3697225533924, 2500.00, 0.10137, -1.0));
  eq(0.2722038980702678, f(436.72578310705217, 2114.3697225533924, 2550.00, 0.10137, -1.0));
  eq(0.44277864026461183, f(4.1066208965498605, 2122.082345327525, 1500.00, 0.178082, -1.0));
  eq(0.423951716475935, f(5.155978570361127, 2122.082345327525, 1550.00, 0.178082, -1.0));
  eq(0.40591493696532394, f(6.518125550789214, 2122.082345327525, 1600.00, 0.178082, -1.0));
  eq(0.3910147948483073, f(8.52603109971654, 2122.082345327525, 1650.00, 0.178082, -1.0));
  eq(0.3790519218852293, f(11.452124613228726, 2122.082345327525, 1700.00, 0.178082, -1.0));
  eq(0.3658033084794829, f(15.054246628138555, 2122.082345327525, 1750.00, 0.178082, -1.0));
  eq(0.35079480824274817, f(19.352577099711628, 2122.082345327525, 1800.00, 0.178082, -1.0));
  eq(0.332831586817996, f(24.185676385823133, 2122.082345327525, 1850.00, 0.178082, -1.0));
  eq(0.31708050087263157, f(30.79461173530755, 2122.082345327525, 1900.00, 0.178082, -1.0));
  eq(0.30348887477724296, f(39.74442189560172, 2122.082345327525, 1950.00, 0.178082, -1.0));
  eq(0.2911200130647552, f(51.408436039119266, 2122.082345327525, 2000.00, 0.178082, -1.0));
  eq(0.2780135269197088, f(65.73620427769617, 2122.082345327525, 2050.00, 0.178082, -1.0));
  eq(0.254268449469205, f(79.71082329912505, 2122.082345327525, 2100.00, 0.178082, -1.0));
  eq(0.23505715618594225, f(99.18448013043029, 2122.082345327525, 2150.00, 0.178082, -1.0));
  eq(0.2290724085266151, f(127.9913662720761, 2122.082345327525, 2200.00, 0.178082, -1.0));
  eq(0.22410062312404985, f(161.69189156563024, 2122.082345327525, 2250.00, 0.178082, -1.0));
  eq(0.2194517986077905, f(199.63020746496065, 2122.082345327525, 2300.00, 0.178082, -1.0));
  eq(0.2149837094034084, f(241.1504654239353, 2122.082345327525, 2350.00, 0.178082, -1.0));
  eq(0.21354072419231343, f(285.9499661135701, 2122.082345327525, 2400.00, 0.178082, -1.0));
  eq(0.2093643138301339, f(332.3134133362891, 2122.082345327525, 2450.00, 0.178082, -1.0));
  eq(0.2100949097173503, f(380.49305653291214, 2122.082345327525, 2500.00, 0.178082, -1.0));
  eq(0.20945394057556008, f(429.32854827566723, 2122.082345327525, 2550.00, 0.178082, -1.0));
  eq(0.2186277951286067, f(478.9712382290464, 2122.082345327525, 2600.00, 0.178082, -1.0));
  eq(0.2169709151643707, f(528.4625785179335, 2122.082345327525, 2650.00, 0.178082, -1.0));
  eq(0.26132095837717184, f(579.2656158990848, 2122.082345327525, 2700.00, 0.178082, -1.0));
  eq(0.24761633018311635, f(628.4542568589878, 2122.082345327525, 2750.00, 0.178082, -1.0));
  eq(0.2527412298159814, f(678.298746365023, 2122.082345327525, 2800.00, 0.178082, -1.0));
  eq(0.41134905571706226, f(21.306793271589243, 2133.480231762442, 1500.00, 0.427397, -1.0));
  eq(0.38634479730938937, f(29.45682087571468, 2133.480231762442, 1600.00, 0.427397, -1.0));
  eq(0.36152899735394906, f(40.19724670074933, 2133.480231762442, 1700.00, 0.427397, -1.0));
  eq(0.3394402792844221, f(55.28913198778564, 2133.480231762442, 1800.00, 0.427397, -1.0));
  eq(0.31986679589203587, f(76.22733104612298, 2133.480231762442, 1900.00, 0.427397, -1.0));
  eq(0.3037283122396698, f(105.25412533971043, 2133.480231762442, 2000.00, 0.427397, -1.0));
  eq(0.25705160323153403, f(125.62919435002404, 2133.480231762442, 2100.00, 0.427397, -1.0));
  eq(0.21971381810933072, f(160.13370820166062, 2133.480231762442, 2200.00, 0.427397, -1.0));
  eq(0.21577168878039, f(225.1496319280382, 2133.480231762442, 2300.00, 0.427397, -1.0));
  eq(0.21254235984957484, f(301.22338205197286, 2133.480231762442, 2400.00, 0.427397, -1.0));
  eq(0.20969412685858838, f(385.897663818452, 2133.480231762442, 2500.00, 0.427397, -1.0));
  eq(0.2078026317658724, f(476.8687633946511, 2133.480231762442, 2600.00, 0.427397, -1.0));
  eq(0.20230730910699307, f(571.2698368997222, 2133.480231762442, 2700.00, 0.427397, -1.0));
  eq(0.20051418828523834, f(668.742528848559, 2133.480231762442, 2800.00, 0.427397, -1.0));
  eq(0.3747977655967484, f(32.762605734107474, 2151.695636331533, 1500.00, 0.676712, -1.0));
  eq(0.3542762338776348, f(43.357995033026185, 2151.695636331533, 1600.00, 0.676712, -1.0));
  eq(0.33528634806426316, f(57.235877487296186, 2151.695636331533, 1700.00, 0.676712, -1.0));
  eq(0.31775819725955773, f(75.3103651148634, 2151.695636331533, 1800.00, 0.676712, -1.0));
  eq(0.3017106893952593, f(98.68254739188998, 2151.695636331533, 1900.00, 0.676712, -1.0));
  eq(0.2874751756073711, f(128.75475525499752, 2151.695636331533, 2000.00, 0.676712, -1.0));
  eq(0.25076261174705244, f(149.99747203567279, 2151.695636331533, 2100.00, 0.676712, -1.0));
  eq(0.22778565317498453, f(187.7045927759423, 2151.695636331533, 2200.00, 0.676712, -1.0));
  eq(0.22204667869563832, f(246.7583066625628, 2151.695636331533, 2300.00, 0.676712, -1.0));
  eq(0.21739801399067926, f(315.52446073985874, 2151.695636331533, 2400.00, 0.676712, -1.0));
  eq(0.21379099783528382, f(392.80847680255994, 2151.695636331533, 2500.00, 0.676712, -1.0));
  eq(0.21051500066557846, f(476.74058026850423, 2151.695636331533, 2600.00, 0.676712, -1.0));
  eq(0.20839399335900793, f(566.1261929324214, 2151.695636331533, 2700.00, 0.676712, -1.0));
  eq(0.20514895061342647, f(658.8358492979602, 2151.695636331533, 2800.00, 0.676712, -1.0));
}
