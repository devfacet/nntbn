#include "nn_accuracy.h"
#include "nn_tensor.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

typedef struct {
    NNTensor *predictions;
    NNTensor *actual;
    NNTensorUnit expected_value;
    NNTensorUnit expected_tolerance;
} TestCase;

const NNTensorUnit default_expected_tolerance = 0.000001f;

void test_nn_accuracy() {
    // See scripts/test/gen/nn_accuracy.py
    TestCase test_cases[] = {
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){1, 2}, false, (const NNTensorUnit[]){0.4722549725289609, 0.5277450274710391}, NULL),
            .actual = nn_tensor_init_NNTensor(2, (const size_t[]){1, 2}, false, (const NNTensorUnit[]){0.0, 1.0}, NULL),
            .expected_value = 1.0,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){1, 2}, false, (const NNTensorUnit[]){0.4260630277170801, 0.57393697228292}, NULL),
            .actual = nn_tensor_init_NNTensor(1, (const size_t[]){1}, false, (const NNTensorUnit[]){1}, NULL),
            .expected_value = 1.0,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){1, 3}, false, (const NNTensorUnit[]){0.21573592713717113, 0.4015099717857861, 0.38275410107704266}, NULL),
            .actual = nn_tensor_init_NNTensor(2, (const size_t[]){1, 3}, false, (const NNTensorUnit[]){0.0, 1.0, 0.0}, NULL),
            .expected_value = 1.0,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){1, 3}, false, (const NNTensorUnit[]){0.4891518179571322, 0.24281339153204934, 0.2680347905108184}, NULL),
            .actual = nn_tensor_init_NNTensor(1, (const size_t[]){1}, false, (const NNTensorUnit[]){2}, NULL),
            .expected_value = 0.0,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){1, 4}, false, (const NNTensorUnit[]){0.2222007327077424, 0.31827322468833813, 0.23639432588444656, 0.2231317167194728}, NULL),
            .actual = nn_tensor_init_NNTensor(2, (const size_t[]){1, 4}, false, (const NNTensorUnit[]){0.0, 0.0, 0.0, 1.0}, NULL),
            .expected_value = 0.0,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){1, 4}, false, (const NNTensorUnit[]){0.1618425611200131, 0.34631244341211465, 0.2601378371446958, 0.23170715832317648}, NULL),
            .actual = nn_tensor_init_NNTensor(1, (const size_t[]){1}, false, (const NNTensorUnit[]){3}, NULL),
            .expected_value = 0.0,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){1, 5}, false, (const NNTensorUnit[]){0.12970697822269728, 0.3081681208743156, 0.1516390499679164, 0.1563193231296622, 0.2541665278054086}, NULL),
            .actual = nn_tensor_init_NNTensor(2, (const size_t[]){1, 5}, false, (const NNTensorUnit[]){0.0, 0.0, 0.0, 0.0, 1.0}, NULL),
            .expected_value = 0.0,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){1, 5}, false, (const NNTensorUnit[]){0.21820744461213473, 0.2089954648204416, 0.16300979842009844, 0.21065842354625852, 0.1991288686010666}, NULL),
            .actual = nn_tensor_init_NNTensor(1, (const size_t[]){1}, false, (const NNTensorUnit[]){0}, NULL),
            .expected_value = 1.0,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){1, 6}, false, (const NNTensorUnit[]){0.22290497580545368, 0.12303735405088066, 0.2018881270490418, 0.1967945944822019, 0.13990900476698925, 0.11546594384543268}, NULL),
            .actual = nn_tensor_init_NNTensor(2, (const size_t[]){1, 6}, false, (const NNTensorUnit[]){1.0, 0.0, 0.0, 0.0, 0.0, 0.0}, NULL),
            .expected_value = 1.0,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){1, 6}, false, (const NNTensorUnit[]){0.13633095584611307, 0.11025306888259634, 0.17562822984705637, 0.09981544254934441, 0.23959765536944783, 0.23837464750544185}, NULL),
            .actual = nn_tensor_init_NNTensor(1, (const size_t[]){1}, false, (const NNTensorUnit[]){1}, NULL),
            .expected_value = 0.0,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){2, 2}, false, (const NNTensorUnit[]){0.3517601052795369, 0.6482398947204631, 0.3691164734520659, 0.6308835265479342}, NULL),
            .actual = nn_tensor_init_NNTensor(2, (const size_t[]){2, 2}, false, (const NNTensorUnit[]){1.0, 0.0, 0.0, 1.0}, NULL),
            .expected_value = 0.5,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){2, 2}, false, (const NNTensorUnit[]){0.6631585916251339, 0.33684140837486604, 0.3019379783579951, 0.6980620216420049}, NULL),
            .actual = nn_tensor_init_NNTensor(1, (const size_t[]){2}, false, (const NNTensorUnit[]){0, 1}, NULL),
            .expected_value = 1.0,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){2, 3}, false, (const NNTensorUnit[]){0.22181460919409407, 0.5117908038937893, 0.2663945869121165, 0.2237276862992539, 0.3817078356333842, 0.39456447806736183}, NULL),
            .actual = nn_tensor_init_NNTensor(2, (const size_t[]){2, 3}, false, (const NNTensorUnit[]){0.0, 0.0, 1.0, 1.0, 0.0, 0.0}, NULL),
            .expected_value = 0.0,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){2, 3}, false, (const NNTensorUnit[]){0.289998915957282, 0.33201373986753835, 0.37798734417517965, 0.2547312132294493, 0.3092069535989049, 0.4360618331716458}, NULL),
            .actual = nn_tensor_init_NNTensor(1, (const size_t[]){2}, false, (const NNTensorUnit[]){1, 0}, NULL),
            .expected_value = 0.0,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){2, 4}, false, (const NNTensorUnit[]){0.3344466376552027, 0.17118938895895386, 0.3022494000158606, 0.19211457336998292, 0.3726927264588274, 0.194040133528421, 0.17654870655850538, 0.25671843345424633}, NULL),
            .actual = nn_tensor_init_NNTensor(2, (const size_t[]){2, 4}, false, (const NNTensorUnit[]){1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}, NULL),
            .expected_value = 0.5,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){2, 4}, false, (const NNTensorUnit[]){0.2850948405326818, 0.24265820943201785, 0.2883844514837149, 0.18386249855158543, 0.26531810446319465, 0.22186787050164453, 0.26284396876102895, 0.24997005627413188}, NULL),
            .actual = nn_tensor_init_NNTensor(1, (const size_t[]){2}, false, (const NNTensorUnit[]){3, 3}, NULL),
            .expected_value = 0.0,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){2, 5}, false, (const NNTensorUnit[]){0.1452463520169489, 0.1629846841086447, 0.35951469171565564, 0.17218759430006592, 0.16006667785868484, 0.16336975673019652, 0.25226605662277274, 0.15306197821307485, 0.23938173939678895, 0.19192046903716692}, NULL),
            .actual = nn_tensor_init_NNTensor(2, (const size_t[]){2, 5}, false, (const NNTensorUnit[]){0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}, NULL),
            .expected_value = 0.0,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){2, 5}, false, (const NNTensorUnit[]){0.1846048452020522, 0.3082342221246576, 0.14878214454277955, 0.12936432550343693, 0.22901446262707362, 0.2784663184193843, 0.2619698602297357, 0.1523557076977058, 0.12502991348667009, 0.1821782001665041}, NULL),
            .actual = nn_tensor_init_NNTensor(1, (const size_t[]){2}, false, (const NNTensorUnit[]){1, 3}, NULL),
            .expected_value = 0.5,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){2, 6}, false, (const NNTensorUnit[]){0.17770364725012375, 0.14452660345239715, 0.11138849160655077, 0.17192778296992364, 0.23115210782090911, 0.16330136690009567, 0.16009315697645277, 0.22020420519561965, 0.1669831029021328, 0.10384610107579811, 0.23268338705290612, 0.11619004679709038}, NULL),
            .actual = nn_tensor_init_NNTensor(2, (const size_t[]){2, 6}, false, (const NNTensorUnit[]){1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0}, NULL),
            .expected_value = 0.0,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){2, 6}, false, (const NNTensorUnit[]){0.15744674034611603, 0.14645334797974952, 0.2699863100468798, 0.13162692446177737, 0.12378110181108265, 0.17070557535439462, 0.21604178877707325, 0.21160431564655252, 0.106232490672748, 0.1470740754328641, 0.10412766888230025, 0.21491966058846163}, NULL),
            .actual = nn_tensor_init_NNTensor(1, (const size_t[]){2}, false, (const NNTensorUnit[]){3, 4}, NULL),
            .expected_value = 0.0,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){3, 2}, false, (const NNTensorUnit[]){0.6956638828320072, 0.3043361171679928, 0.6257434800484648, 0.37425651995153514, 0.3043664435333136, 0.6956335564666863}, NULL),
            .actual = nn_tensor_init_NNTensor(2, (const size_t[]){3, 2}, false, (const NNTensorUnit[]){0.0, 1.0, 0.0, 1.0, 0.0, 1.0}, NULL),
            .expected_value = 0.3333333333333333,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){3, 2}, false, (const NNTensorUnit[]){0.6162853572591995, 0.38371464274080047, 0.5962068265843739, 0.40379317341562626, 0.4013097497198873, 0.5986902502801128}, NULL),
            .actual = nn_tensor_init_NNTensor(1, (const size_t[]){3}, false, (const NNTensorUnit[]){0, 0, 0}, NULL),
            .expected_value = 0.6666666666666666,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){3, 3}, false, (const NNTensorUnit[]){0.19435488912784832, 0.33240265998335955, 0.47324245088879213, 0.22849503303354848, 0.274325396117012, 0.4971795708494396, 0.2590927027732187, 0.40480909327272463, 0.3360982039540567}, NULL),
            .actual = nn_tensor_init_NNTensor(2, (const size_t[]){3, 3}, false, (const NNTensorUnit[]){1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0}, NULL),
            .expected_value = 0.3333333333333333,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){3, 3}, false, (const NNTensorUnit[]){0.4073074722873894, 0.30366930140000714, 0.28902322631260347, 0.25205208454122985, 0.33544691552424455, 0.4125009999345255, 0.3861625165571821, 0.3470069523189316, 0.2668305311238864}, NULL),
            .actual = nn_tensor_init_NNTensor(1, (const size_t[]){3}, false, (const NNTensorUnit[]){1, 1, 1}, NULL),
            .expected_value = 0.0,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){3, 4}, false, (const NNTensorUnit[]){0.16668477063716897, 0.27275703526315714, 0.3381242839219978, 0.22243391017767614, 0.27676100376928675, 0.18363133464393197, 0.28211571484602666, 0.25749194674075454, 0.22313816544695259, 0.21769528382605258, 0.3707952397791245, 0.1883713109478704}, NULL),
            .actual = nn_tensor_init_NNTensor(2, (const size_t[]){3, 4}, false, (const NNTensorUnit[]){1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0}, NULL),
            .expected_value = 0.3333333333333333,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){3, 4}, false, (const NNTensorUnit[]){0.3437449875889564, 0.236243891406123, 0.23251394110949047, 0.18749717989543016, 0.32053587858884797, 0.2114859790316651, 0.30126695034147966, 0.16671119203800727, 0.1317780565053611, 0.31659134607349876, 0.3166557441762117, 0.2349748532449285}, NULL),
            .actual = nn_tensor_init_NNTensor(1, (const size_t[]){3}, false, (const NNTensorUnit[]){1, 0, 3}, NULL),
            .expected_value = 0.3333333333333333,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){3, 5}, false, (const NNTensorUnit[]){0.2230013316843514, 0.17695421445113332, 0.2536592482374963, 0.16626115996937355, 0.1801240456576454, 0.22221089871092864, 0.19446726632850858, 0.24607010408435628, 0.15299738208844887, 0.1842543487877576, 0.17118299316870986, 0.16661325911729344, 0.150176470539909, 0.22227408799866807, 0.28975318917541965}, NULL),
            .actual = nn_tensor_init_NNTensor(2, (const size_t[]){3, 5}, false, (const NNTensorUnit[]){0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0}, NULL),
            .expected_value = 0.0,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){3, 5}, false, (const NNTensorUnit[]){0.17045927972523198, 0.2014285966509104, 0.14680134885127613, 0.27889388142111726, 0.20241689335146418, 0.1398751334826821, 0.17648384130306297, 0.2504514048531559, 0.15389671733466548, 0.27929290302643356, 0.22826249177619515, 0.15548389303545446, 0.17478804354504812, 0.2328703885697761, 0.20859518307352606}, NULL),
            .actual = nn_tensor_init_NNTensor(1, (const size_t[]){3}, false, (const NNTensorUnit[]){2, 3, 0}, NULL),
            .expected_value = 0.0,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){3, 6}, false, (const NNTensorUnit[]){0.15805729564017373, 0.23252075083427606, 0.19150179518649899, 0.21512925295258994, 0.1095996332717742, 0.09319127211468711, 0.1185221252388811, 0.24933525845299173, 0.15285962851306562, 0.17312754548917672, 0.17827121841795424, 0.12788422388793058, 0.1889530811274252, 0.1326566274045956, 0.1881967849270929, 0.12384380227112143, 0.24120898442995056, 0.12514071983981434}, NULL),
            .actual = nn_tensor_init_NNTensor(2, (const size_t[]){3, 6}, false, (const NNTensorUnit[]){0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}, NULL),
            .expected_value = 0.3333333333333333,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){3, 6}, false, (const NNTensorUnit[]){0.10512784946276008, 0.23698664768116895, 0.1644295206704906, 0.18677233618341985, 0.10731048142856926, 0.1993731645735912, 0.15939387495811394, 0.19029143737153068, 0.12175433159216219, 0.1399006692005241, 0.1234155430257846, 0.26524414385188455, 0.15093083596624662, 0.18111342178532477, 0.16740138859013265, 0.16564748709605634, 0.15274890557894918, 0.18215796098329046}, NULL),
            .actual = nn_tensor_init_NNTensor(1, (const size_t[]){3}, false, (const NNTensorUnit[]){4, 1, 1}, NULL),
            .expected_value = 0.0,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){4, 2}, false, (const NNTensorUnit[]){0.3459704725388397, 0.6540295274611603, 0.3651981653910368, 0.6348018346089631, 0.5464900422025236, 0.4535099577974764, 0.466507766783521, 0.5334922332164789}, NULL),
            .actual = nn_tensor_init_NNTensor(2, (const size_t[]){4, 2}, false, (const NNTensorUnit[]){0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0}, NULL),
            .expected_value = 0.75,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){4, 2}, false, (const NNTensorUnit[]){0.6953832238798521, 0.3046167761201479, 0.44642652617105627, 0.5535734738289437, 0.6599032244747773, 0.34009677552522277, 0.4919859418627378, 0.5080140581372622}, NULL),
            .actual = nn_tensor_init_NNTensor(1, (const size_t[]){4}, false, (const NNTensorUnit[]){0, 1, 1, 0}, NULL),
            .expected_value = 0.5,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){4, 3}, false, (const NNTensorUnit[]){0.3232570575481441, 0.220195988807561, 0.45654695364429493, 0.359760578691911, 0.23479856141655656, 0.4054408598915325, 0.36942064383545725, 0.33520431047756755, 0.2953750456869752, 0.3171034379095906, 0.29257353177406464, 0.3903230303163448}, NULL),
            .actual = nn_tensor_init_NNTensor(2, (const size_t[]){4, 3}, false, (const NNTensorUnit[]){0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0}, NULL),
            .expected_value = 0.25,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){4, 3}, false, (const NNTensorUnit[]){0.4546690027286087, 0.22356176558156288, 0.32176923168982824, 0.19013783044402202, 0.4199715526069224, 0.3898906169490555, 0.5191129994843476, 0.265606278029094, 0.21528072248655825, 0.2379496290469206, 0.35815952859731776, 0.40389084235576167}, NULL),
            .actual = nn_tensor_init_NNTensor(1, (const size_t[]){4}, false, (const NNTensorUnit[]){0, 0, 2, 2}, NULL),
            .expected_value = 0.5,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){4, 4}, false, (const NNTensorUnit[]){0.21565055332414373, 0.23093624797059722, 0.2411975017027301, 0.312215697002529, 0.1708772906459721, 0.23112976713572578, 0.2626517593219415, 0.33534118289636056, 0.19749345038312888, 0.37384746940173663, 0.20856749560077234, 0.2200915846143623, 0.3096275367268565, 0.22497905330127163, 0.23689305400541566, 0.2285003559664562}, NULL),
            .actual = nn_tensor_init_NNTensor(2, (const size_t[]){4, 4}, false, (const NNTensorUnit[]){0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}, NULL),
            .expected_value = 0.5,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){4, 4}, false, (const NNTensorUnit[]){0.33851917394777464, 0.16212314142953696, 0.31512857026613017, 0.1842291143565582, 0.22666977492651125, 0.2546439081317649, 0.33432010855295996, 0.18436620838876394, 0.15536107741035748, 0.22608817663011355, 0.3256297570167403, 0.2929209889427888, 0.16315511470025262, 0.24666866978526603, 0.22176108888859233, 0.36841512662588904}, NULL),
            .actual = nn_tensor_init_NNTensor(1, (const size_t[]){4}, false, (const NNTensorUnit[]){0, 1, 0, 3}, NULL),
            .expected_value = 0.5,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){4, 5}, false, (const NNTensorUnit[]){0.20151334016619699, 0.228311090781654, 0.22823348342186484, 0.13998068375675668, 0.20196140187352757, 0.22619438043271245, 0.22934468423055243, 0.13702833617394347, 0.18483278222594424, 0.2225998169368473, 0.21387939992451177, 0.14126269875094197, 0.2669077747453805, 0.1758210408246474, 0.20212908575451835, 0.19366494214368626, 0.1748209916460346, 0.20726063063138142, 0.2799390380731366, 0.1443143975057611}, NULL),
            .actual = nn_tensor_init_NNTensor(2, (const size_t[]){4, 5}, false, (const NNTensorUnit[]){0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0}, NULL),
            .expected_value = 0.0,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){4, 5}, false, (const NNTensorUnit[]){0.19695295376289335, 0.30856326625764885, 0.19803383353521145, 0.14085792861839772, 0.1555920178258486, 0.2773223755890561, 0.16686774598779291, 0.21653540977697072, 0.19732913763056528, 0.14194533101561504, 0.24955288201179196, 0.16318802421264522, 0.2721972256621927, 0.1842185457811171, 0.13084332233225315, 0.19288209733253547, 0.15348217276648213, 0.30378342725546825, 0.2015638465483161, 0.14828845609719787}, NULL),
            .actual = nn_tensor_init_NNTensor(1, (const size_t[]){4}, false, (const NNTensorUnit[]){2, 4, 2, 1}, NULL),
            .expected_value = 0.25,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){4, 6}, false, (const NNTensorUnit[]){0.28922351865893076, 0.17162599442256635, 0.12214268380315761, 0.14556869659908334, 0.1498506455898455, 0.12158846092641658, 0.09641916032034643, 0.21518884114667952, 0.24940796916821398, 0.20411675367406287, 0.10694895553324313, 0.12791832015745416, 0.15161193630417427, 0.14468919300601674, 0.12882726169887448, 0.2570235159716976, 0.13836606113510963, 0.1794820318841273, 0.15926719823293237, 0.12910861145352398, 0.22920165327802397, 0.11547343740559775, 0.2142052815354945, 0.15274381809442755}, NULL),
            .actual = nn_tensor_init_NNTensor(2, (const size_t[]){4, 6}, false, (const NNTensorUnit[]){0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0}, NULL),
            .expected_value = 0.0,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){4, 6}, false, (const NNTensorUnit[]){0.21054187179494893, 0.17017696263776502, 0.17061017174345544, 0.10686465462222841, 0.1754433997379026, 0.16636293946369965, 0.1787384318637779, 0.136332307398377, 0.1301419815680993, 0.14353869600522648, 0.1706794048753843, 0.24056917828913488, 0.11872026871020125, 0.1376326574662959, 0.22920107628317918, 0.14300257456209772, 0.2633374912102036, 0.10810593176802229, 0.0981043881841555, 0.23608242209365668, 0.23460273271687404, 0.16957365446846753, 0.1001454726620575, 0.16149132987478876}, NULL),
            .actual = nn_tensor_init_NNTensor(1, (const size_t[]){4}, false, (const NNTensorUnit[]){5, 2, 5, 0}, NULL),
            .expected_value = 0.0,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){5, 2}, false, (const NNTensorUnit[]){0.43954612078194316, 0.5604538792180568, 0.565503648770819, 0.43449635122918095, 0.47153918075324003, 0.5284608192467599, 0.629769873090171, 0.37023012690982904, 0.5738710961985267, 0.4261289038014733}, NULL),
            .actual = nn_tensor_init_NNTensor(2, (const size_t[]){5, 2}, false, (const NNTensorUnit[]){1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0}, NULL),
            .expected_value = 0.4,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){5, 2}, false, (const NNTensorUnit[]){0.6579336913818188, 0.34206630861818116, 0.583078565400579, 0.4169214345994209, 0.5681718746542153, 0.43182812534578474, 0.5216289188766539, 0.47837108112334614, 0.43202232853179545, 0.5679776714682047}, NULL),
            .actual = nn_tensor_init_NNTensor(1, (const size_t[]){5}, false, (const NNTensorUnit[]){0, 0, 0, 1, 0}, NULL),
            .expected_value = 0.6,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){5, 3}, false, (const NNTensorUnit[]){0.2626139162694406, 0.505200165831961, 0.23218591789859833, 0.3473810268453183, 0.32114222303634776, 0.331476750118334, 0.3180077765779295, 0.292763027172745, 0.3892291962493255, 0.21579782704558553, 0.4378780112742304, 0.34632416168018404, 0.29642686936364654, 0.33824182141383785, 0.36533130922251555}, NULL),
            .actual = nn_tensor_init_NNTensor(2, (const size_t[]){5, 3}, false, (const NNTensorUnit[]){0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0}, NULL),
            .expected_value = 0.2,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){5, 3}, false, (const NNTensorUnit[]){0.25778432598304274, 0.4316073216891931, 0.31060835232776407, 0.30021406248175775, 0.2561943557283585, 0.44359158178988384, 0.36090575838815214, 0.1994852512570285, 0.43960899035481943, 0.46471944945990273, 0.22933093696855902, 0.3059496135715382, 0.31561323978156036, 0.39018817325499344, 0.2941985869634462}, NULL),
            .actual = nn_tensor_init_NNTensor(1, (const size_t[]){5}, false, (const NNTensorUnit[]){2, 1, 1, 2, 2}, NULL),
            .expected_value = 0.0,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){5, 4}, false, (const NNTensorUnit[]){0.13054520099261183, 0.2723670877319736, 0.2924699267618827, 0.3046177845135319, 0.23372615585569267, 0.18418062623521084, 0.32061395895993233, 0.2614792589491642, 0.20382577009327402, 0.3204983209168751, 0.2968437543080178, 0.17883215468183314, 0.20680630667756816, 0.20001729096036322, 0.3451182150066244, 0.24805818735544427, 0.3185036688860527, 0.21165236851997143, 0.1490846985835583, 0.32075926401041754}, NULL),
            .actual = nn_tensor_init_NNTensor(2, (const size_t[]){5, 4}, false, (const NNTensorUnit[]){0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0}, NULL),
            .expected_value = 0.2,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){5, 4}, false, (const NNTensorUnit[]){0.152865804942464, 0.3262796467875006, 0.22561387328238625, 0.2952406749876491, 0.3324428288309149, 0.17719759383691414, 0.18698298889904766, 0.3033765884331233, 0.3348469841901128, 0.20665287491524828, 0.23973591244223785, 0.21876422845240115, 0.18704772358523258, 0.2647830761313281, 0.35862207794237466, 0.18954712234106472, 0.2849852370991467, 0.3285922318399031, 0.2488708180215847, 0.13755171303936556}, NULL),
            .actual = nn_tensor_init_NNTensor(1, (const size_t[]){5}, false, (const NNTensorUnit[]){3, 3, 0, 2, 3}, NULL),
            .expected_value = 0.4,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){5, 5}, false, (const NNTensorUnit[]){0.16906159063883014, 0.21883270629050922, 0.2789952038491681, 0.1498879203232753, 0.18322257889821716, 0.28239803130382635, 0.22540434633148615, 0.1844092532644384, 0.15014777221991712, 0.157640596880332, 0.1763934473324373, 0.28501936367962066, 0.11804646817879635, 0.20889611373258843, 0.2116446070765571, 0.15246712501177884, 0.13055635521303516, 0.21062872010895692, 0.18241989097761419, 0.32392790868861493, 0.14906401549635953, 0.1966917533265738, 0.17075671082682664, 0.21822529723222406, 0.265262223118016}, NULL),
            .actual = nn_tensor_init_NNTensor(2, (const size_t[]){5, 5}, false, (const NNTensorUnit[]){0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0}, NULL),
            .expected_value = 0.2,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){5, 5}, false, (const NNTensorUnit[]){0.2104420305855768, 0.20492712547336023, 0.29705401788038294, 0.14473465136085564, 0.14284217469982435, 0.12954625792250282, 0.2973931716227574, 0.1749884680521817, 0.17020683768638215, 0.227865264716176, 0.16411478862387846, 0.2103608815610703, 0.22637674731815174, 0.23444789970538057, 0.16469968279151884, 0.19095181180572687, 0.25144237442412404, 0.2033462615384212, 0.1559479505394696, 0.19831160169225825, 0.19005841192726597, 0.17126404728535713, 0.2047400895978433, 0.23419333533318165, 0.1997441158563519}, NULL),
            .actual = nn_tensor_init_NNTensor(1, (const size_t[]){5}, false, (const NNTensorUnit[]){4, 1, 0, 2, 3}, NULL),
            .expected_value = 0.4,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){5, 6}, false, (const NNTensorUnit[]){0.1502148206377633, 0.10118882387242956, 0.1392023860157386, 0.15504241868864196, 0.22980512596290695, 0.22454642482251957, 0.13525159917356722, 0.26477433741218503, 0.13876792788260942, 0.16357900597491576, 0.1500897329537055, 0.1475373966030172, 0.1936372861209115, 0.13089375376613313, 0.18812667516752382, 0.20047726925866444, 0.12107211361925731, 0.16579290206750974, 0.1571681735146479, 0.10528907413157662, 0.24130771450662444, 0.2546542328264516, 0.12986671405273117, 0.11171409096796817, 0.20502743061087592, 0.10923423735901397, 0.12095870123048626, 0.2143102856253843, 0.1692424134106782, 0.1812269317635613}, NULL),
            .actual = nn_tensor_init_NNTensor(2, (const size_t[]){5, 6}, false, (const NNTensorUnit[]){0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0}, NULL),
            .expected_value = 0.2,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){5, 6}, false, (const NNTensorUnit[]){0.26429384903422776, 0.16994550760042376, 0.15063256713428794, 0.13089186934457242, 0.17352037111652036, 0.11071583576996778, 0.1913944925341013, 0.15732124633064415, 0.13590281612668495, 0.2334206308190439, 0.1116152807691593, 0.1703455334203664, 0.17940763340006483, 0.1824904206158282, 0.15790804720825188, 0.1899387751628373, 0.1897738565266964, 0.10048126708632137, 0.17926030312360214, 0.16402540414530792, 0.16690263945711767, 0.11487336694423175, 0.19424596511567435, 0.18069232121406598, 0.09217276091882194, 0.23127726865605827, 0.20591631554098566, 0.11090128314485159, 0.1839190360066395, 0.17581333573264307}, NULL),
            .actual = nn_tensor_init_NNTensor(1, (const size_t[]){5}, false, (const NNTensorUnit[]){5, 5, 3, 0, 1}, NULL),
            .expected_value = 0.4,
            .expected_tolerance = default_expected_tolerance,
        },
    };

    const int n_cases = sizeof(test_cases) / sizeof(test_cases[0]);
    for (int i = 0; i < n_cases; i++) {
        TestCase tc = test_cases[i];

        NNError error = {0};
        const NNTensorUnit accuracy = nn_accuracy(tc.predictions, tc.actual, &error);
        assert(isnan(accuracy) == false);
        assert(error.code == NN_ERROR_NONE);
        assert(fabs(accuracy - tc.expected_value) < tc.expected_tolerance);
        printf("passed: %s case=%d\n", __func__, i + 1);
    }
}
