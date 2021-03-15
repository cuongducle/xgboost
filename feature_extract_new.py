import re
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict

class Vectorizator():
    def __init__(self):
        self.dauCharList = {"á", "à", "ả", "ã", "ạ", "ă", "ắ", "ằ", "ẳ", "ẵ", "ặ", "â", "ấ", "ầ", "ẩ", "ẫ", "ậ", "é", "è", "ẻ", "ẽ", "ẹ", "ê", "ế", "ề", "ể", "ễ", "ệ", "í", "ì", "ỉ","ĩ", "ị", "ó", "ò", "ỏ", "õ", "ọ", "ô", "ố", "ồ", "ổ", "ỗ", "ộ", "ơ", "ớ", "ờ", "ở", "ỡ", "ợ", "ú", "ù", "ủ", "ũ", "ụ", "ư", "ứ", "ừ", "ử", "ữ", "ự", "ý", "ỳ", "ỷ", "ỹ", "ỵ"}
        self.nguyenAmCharList = {"a","e","i","o","u","y","á","à","ả","ã","ạ","ă","ắ","ằ","ẳ","ẵ","ặ","â","ấ","ầ","ẩ","ẫ","ậ","é","è","ẻ","ẽ","ẹ","ê","ế","ề","ể","ễ","ệ","í","ì","ỉ","ĩ","ị","ó","ò","ỏ","õ","ọ","ô","ố","ồ","ổ","ỗ","ộ","ơ","ớ","ờ","ở","ỡ","ợ","ú","ù","ủ","ũ","ụ","ư","ứ","ừ","ử","ữ","ự","ý","ỳ","ỷ","ỹ","ỵ"}
        self.phuAmCharList = {"b","c","d","f","g","h","j","k","l","m","n","p","q","r","s","t","v","w","x","z","đ"}
        self.nuocNgoaiCharList = {"f","j","z","w"}
        self.upperCharList = {"A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","Ắ","Ằ","Ẳ","Ẵ","Ặ","Ă","Ấ","Ầ","Ẩ","Ẫ","Ậ","Â","Á","À","Ã","Ả","Ạ","Đ","Ế","Ề","Ể","Ễ","Ệ","Ê","É","È","Ẻ","Ẽ","Ẹ","Í","Ì","Ỉ","Ĩ","Ị","Ố","Ồ","Ổ","Ỗ","Ộ","Ô","Ớ","Ờ","Ở","Ỡ","Ợ","Ơ","Ó","Ò","Õ","Ỏ","Ọ","Ứ","Ừ","Ử","Ữ","Ự","Ư","Ú","Ù","Ủ","Ũ","Ụ","Ý","Ỳ","Ỷ","Ỹ","Ỵ"}
        self.lowerCharList = {"a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","á","à","ả","ã","ạ","ă","ắ","ằ","ẳ","ẵ","ặ","â","ấ","ầ","ẩ","ẫ","ậ","é","è","ẻ","ẽ","ẹ","ê","ế","ề","ể","ễ","ệ","í","ì","ỉ","ĩ","ị","ó","ò","ỏ","õ","ọ","ô","ố","ồ","ổ","ỗ","ộ","ơ","ớ","ờ","ở","ỡ","ợ","ú","ù","ủ","ũ","ụ","ư","ứ","ừ","ử","ữ","ự","ý","ỳ","ỷ","ỹ","ỵ","đ"}
        self.numCharList = {"0","1","2","3","4","5","6","7","8","9"}
        self.puncDateCharList = {"-","/"}
        self.puncHourCharList = {":"}
        self.abbrevCharList = {"@","."}
        self.otherCharList = {"`","~","!","#","^","&","*","(",")","_","-","<",">"}
        self.mathCharList = {"+","-","*","/",":","="}
        self.measureUnitCharList = {"m/s","km/s","cm/s","l/m","l/km","km/h","kg/l","g/l"}
        self.currencyCharList = {"$","yên","nhân dân tệ","đồng","euro","tệ","đô la","au","đô-la","rs","kn","aud","irs","usd","hk$","can$","bt"}
        self.measureCharList = {"p","ha","ms","mcg","kcal","km","dm","cm","mm","nm","m2","km2","m3","km3","kg","mg","°","oc","of","khz","mhz","ghz","hz","khz","mhz","ghz","hz","pa","kj","pa","kwh","kw","mah","mW","mw","dl","cl","ml","bq","ft","m","l","W"}
        self.romanCharList = {"i","ii","iii","I","II","III","IV","V","VI","VII","VIII","IX","X","XI","XII","XIII","XIV","XV","XVI","XVII","XVIII","XIX","XX","XXI","XXII","XXIII","XXIV","XXV","XXVI","XXVII","XXVIII","XXIX","XXX","XXXI","XXXII","XXXIII","XXXIV","XXXV","XXXVI","XXXVII","XXXVIII","XXXIX","XL","XLI","XLII","XLIII","XLIV","XLV","XLVI","XLVII","XLVIII","XLIX","L","LI","LII","LIII","LIV","LV","LVI","LVII","LVIII","LIX","LX","LXI","LXII","LXIII","LXIV","LXV","LXVI","LXVII","LXVIII","LXIX","LXX","LXXI","LXXII","LXXIII","LXXIV","LXXV","LXXVI","LXXVII","LXXVIII","LXXIX","LXXX","LXXXI","LXXXII","LXXXIII","LXXXIV","LXXXV","LXXXVI","LXXXVII","LXXXVIII","LXXXIX","XC","XCI","XCII","XCIII","XCIV","XCV","XCVI","XCVII","XCVIII","XCIX","C","CC","CCC","CD","D","DC","DCC","DCCC","CM","M"}
        self.hourTextList = {"am","pm","h"}
        self.minuteTextList = {"p"}
        self.secTextList = {"s"}
        self.dateTextList = {"ngày","sáng","trưa","chiều","tối"}
        self.monthTextList = {"tháng"}
        self.quyTextList = {"quý"}
        self.yearTextList = {"năm"}
        self.timeTextList = {"lúc","vào","hồi","khoảng"}
        self.fractionTextList = {"chiếm","khoảng","trên","dưới","gần"}
        self.scoreTextList = {"tỷ số","thắng","thua","hòa"}
        self.romanTextList = {"phần","giai đoạn","cấp","đại hội","kỳ","khóa"}
        self.abbrList = {"gl","hqđt","tba","tth","tns","đh","gdđh","nha","gdtx","tnnt","dagtđt","xsmn","bgh","đcs","sp","xl","gpxd","hvannd","shnn","csđt","tvhk","yb","tdv","sđtla","ptn","ptthttđt","cskh","tphn","vlxd","tb","kntc","ptnn","lthđt","pctnxh","sg","lsq","đbtn","atnđ","hg","thvl","ace","mk","bgd","qcgnhh","pclbtw","nk","đn","dnvvn","cptd","kđt","sgkv","bhtn","csđttpmt","tdttqg","nltt","đcstq","pctubnd","cntn","nsnn","la","ntm","đhcn","nkkn","tthlttqg","gvc","tbmmn","qt","ubnd","pl","gđ","bđhn","gplx","kk","ptlc","ubcm","cmnd","gptm","nt","đhktcn","đhhhvn","svđv","xd","catp","klđt","klqg","đcsvn","qsd","tnhhmtv","hctc","cqld","tacn","dtxd","kcb","ha","ytdp","đttx","kkbhxh","kđcl","dtht","xhh","gtgt","tsdđ","tbktvn","yt","brvt","sxkd","đbvn","blđ","tbxh","cnch","pnkb","tbtt","tntn","kg","vp","nhctcn","đkhk","hđcdgscs","đna","tcdl","sk","th","đvcnt","pctt","tbn","lcđ","gv","slna","shtt","cntt","khcn","kdnpn","chdc","ubmttq","cspk","thkts","cskt","ttvh","spkt","bđqg","hlg","ps","xskt","đb","ktxh","ttgdck","csđn","xđgn","ctsn","bhx","gmd","ghn","nbcl","đttp","sgtvt","ctn","lđlđvn","bbcvt","pctpcnc","khxhnv","pclbmn","đbdtts","đhgtvt","đhđd","đtnn","mtv","thcb","gcnsk","hmnđ","kn","cn","q","kđclgd","lc","qđxpvphc","dthcp","tkbvtc","mtxt","ttxh","đtvn","ms","qc","tvn","đs","chnd","tbktsg","nvhtn","tandtc","gdnn","xdtm","pctttkcn","đtla","đhbk","dnnn","hđt","cm","tcn","hk","cbcnv","ptth","bchtư","tq","ubck","xttm","uvbch","hn","dvvl","tctt","nmđ","tthtsv","nd","stt","hđtđ","tcnn","ttksgt","qh","hsv","tmđt","đttm","pk","sgd","tm","đhnnhn","cc","tbcn","bch","ncs","nhnt","tntp","ghtk","tx","bp","xnk","sđnđ","hkhkt","vltk","cmtt","cty","bql","bctc","bxh","gd","cnxh","nhcthp","thcsnđ","atvstp","thptdtnt","cpxdsxtm","đkkd","bl","btv","ttvn","đvht","ctcc","kdl","hđđb","khđt","tvqh","gcnqsdđ","tttt","qshnqsdđ","tc","lđtb","khkths","đhcđ","bcnckt","tdp","đttxqm","ctđl","ttdv","vđv","tnđt","đđ","cv","nxbqđnd","đhnt","đsq","hs","nsx","sx","blhs","hlv","đlđt","anđt","bt","nncđdc","sl","lhtn","tsn","ntk","qtkd","ut","sd","syll","bcđ","hsđkdt","antđ","khktnn","csvc","kdhđc","ghpgvn","bcđts","svhs","skss","sggp","GĐKT","hv","chxhcnvn","vl","cnkt","đhkhxh","ttyt","hl","đta","cnh","ptsx","xn","tpct","bcđđmđh","vck","ql","nst","mtđt","kl","UBQLV","qctđhn","đct","cđv","ykvn","lmlm","gcnqshnơ","ag","đhbkhn","ctch","chlb","mssv","bvđk","hqhp","đtv","vh","đhdl","đtlt","lđ","tcqt","tư","đbdt","gtvt","đbtsxb","ctth","hđkt","pcgdbth","vpls","skhđt","xh","gtmt","btbnn","khhgđ","cntttt","pp","vpđd","hb","pghh","kkt","gstskh","bs","tđbkvn","bđn","sxtm","htqt","ubndtp","ub","vđqg","hđtt","ltqđtd","ts","nv","chk","tnkq","hđxx","ntđ","ctđ","gsts","htx","pcsddte","khkthn","pv","blv","rlcd","HKVN","hlhtnvn","vksndtc","bbt","ktx","nhctvn","cnqsdđ","ttvgt","xklđ","vhttdl","kdc","tchq","pcmttp","đhbc","sdđ","hht","pcmt","hvch","vksnd","csvn","dv","hhc","pcccchcn","cttt","gt","tmcp","bcvt","pct","tccn","pt","kt","khktnnmn","đmn","tncn","qhtd","đsvn","xnql","tthc","gđđh","hđnt","thcs","ubtvqh","nh","thads","csqlhc","tctđô","tpcđ","đg","mn","nlđ","qshnơ","tmn","ptt","cb","dvtt","hp","uv","tp","qlđkxlx","hvcnbcvt","bhđc","cx","svđ","ccv","lh","cnv","ncl","qchq","st","ttct","hđcdgsn","bks","bh","ct","hd","skhcnmt","nhđt","ttth","svnckh","hhtg","đbsh","lđbđ","bc","pgđ","vn","mt","bcđpctntư","mmt","ndo","tcmn","hvbcvt","tnsvvn","tw","bhxh","hđcd","lnst","ttk","đhqghn","dd","ttbdvh","nhnnvn","antt","pclb","cap","đvttn","ctck","hkddvn","tv","bhyt","cscđ","hvnclc","nckh","qcdccs","svtn","tnk","tk","tcvsg","tcttmsg","đrl","pctn","đc","ubpl","ccspctpmt","cnkqt","qn","nvh","tnmt","gcnqsh","nxbgd","tndn","tkcn","đhn","đd","đbp","ktqdhn","na","qđnd","ch","nxbtn","ttgt","thx","cskv","gcnqshn","thktsmđ","nmct","đspl","vpcp","svvn","đtvhd","h","qckdtxd","tđ","gpmb","hcb","lđbđvn","vcgt","lhs","nhtmnn","cmt","ubcknn","ctxh","ptnt","csđttp","ktt","kđtm","cđsp","lb","pslđ","ttqlkt","hngđ","lntt","qg","ttttn","vhxh","lhq","tpdn","nxb","atgt","khxh","mxh","vs","plo","cpttt","bd","đbqh","hđh","qlđt","htxdvnn","tcmr","nhtw","ttdvvl","ubgstc","hvn","hsgqg","qlda","đhkh","tg","cchc","pgs","hnv","tctd","TTCP","cbqlgd","gvtn","hctl","khtn","qsdđ","dnvn","gvmn","thtt","tdtt","hđcdgs","bhl","đđv","snn","tđc","tcbvntd","đtnđ","hphn","gk","khls","sđk","mbh","xdhk","đhclc","bxl","lmht","nqd","đk","tnxk","pcgdth","bgđ","tpcn","hlbđ","tths","tand","ttn","ttatgt","nđvn","nsưt","đhkhtnhn","ko","nsđp","nxbvhtt","gdgt","vks","caq","cssx","clb","cahn","tgdđ","plvn","ktm","tmdv","ktđt","csdc","dl","bx","csdl","ptdtbt","khkd","vthkcc","gcn","nkbv","tct","hagl","attt","bg","gđtla","shtd","vhnt","gdđt","cncn","đhkt","cbpg","dnt","pm","tnln","tn","ktv","nhnn","kv","rlc","kcx","ntbd","sgtt","tbt","ktst","kst","đhbktphcm","pgd","nsnd","ctcp","phhs","pgsts","đdv","CQĐT","đtxdcb","ss","đt","pcgdthcs","hhvn","dcch","hlqg","bđbp","pcbl","nxbkhxh","pcbth","ttdd","uvbct","csvl","đv","ls","nhtm","hcvl","cp","npp","đl","lhpn","tncs","đtb","annd","xhcn","tnvn","bqlda","đbscl","csg","mttqvn","nhđá","tttm","dlbc","kdđv","hh","qk","gddt","hđđg","nđ","tcvn","dn","cttnhh","ttnn","hvkhqs","cđ","cpxd","xsmb","scn","tnsv","vhgdttnnđ","bđ","pcgd","sdđnn","qcvn","TQ","nhnnptntvn","gvg","nn","thdl","nvơnn","noxh","ccb","qb","mst","vđvqg","vnđ","đhqg\00đại học sư phạm","btc","ttck","nhm","cshs","ubtdtt","gcnqsdđơ","chdcnd","gvcn","sgk","ưcv","hnht","bvmt","hđndtp","ttbđs","vđ","vđtg","bgdđt","tpo"," CLC","ttbvqtg","ubkt","sn","nhcsxh","sh","nđt","thpt","dntn","cbcc","bts","thcn","py","đhktqd","vv","nhnnptnt","qđ","ubkttw","đkkh","đhbkhcm","cand","ttks","ttxvn","hcm","đbdtin","dvtmcm","atttgt","bvtv","hsbc","hđct","tvtw","hdv","thgt","hkhktvn","da","sđt","tmxd","qltt","nkt","vhtt","nq","hđlđ","pctp","nnptnt","ksnd","hđba","gplhđb","cbnvlđ","ca","ccnv","tnv","gcnqsd","ptv","gp","gpnk","lđlđ","đvtn","htxnn","skhcn","nltd","hvncl","ll","cmcn","hđnd","đtdđ","cvpm","nlmt","qbltd","kđs","nơxh","cbcs","kbnn","cbnv","cnvc","vhgdtntn","tvv","btvh","hc","xk","gcq","csgt","hspt","nctnt","ptcs","lct","đkvn","pttm","xx","qsmt","tnt","pcthcs","hy","hq","tckh","ttđm","kp","hđts","bchqs","đhđb","pcc","nhtmcp","hnd","ck","ksv","stda","hm","htcđ","smđh","hsg","tbd","nls","nc","ctv","hđqt","attp","kts","llvt","ths","tttđb","bđs","lđtbxh","ttdl","csht","vnclc","đkvđ","đkđt","hđ","sxh","đtđ","đvưt","cttc","đkdt","bđkh","đtqg","pccc","chxhcn","ntd","vt","ptdt","sđh","đhđcqt","kh","bcđt","ubqgtkcn","xdcb","ttton","ksndtc","vc","csxh","ks","kđtntl","nctt","bcđqg","ttcn","csđtlx","tgpl","gđt","vb","qsdđơ","kttv","ns","nknv","BTTN","hktt","bv","hcđ","pn","tnhh","pc","tckt","nhtmqd","qld","đhhb","gs","qckt","hsgqt","lhp","bchtw","dhs","ttcp","nl","ht","bn","td","kdcn","cnhhđh","ndt","ttgdtx","chxh","ttđt","hđtv","vd","đhđcđ","dt","ntls","tgđ","cnsh","hđcdgsnn","cs","hcv","sbd","đhđn","hnbvn","hssv","kcn","kdn","thvn","tngt","tnđh","qsqp","khkt","gstt","pcbltkcn","ctcn","tccs","bgk","mttq","sv","ttytdp","dvtm","CTCK","mhx","dtkv","tphcm","pttt","ctclqg","kcbcnn","gcnkqt","lllđ","sxtmdv","thqg","tchl","gđtt","đhct","tskh","hvnh","đhkhtn","bđvn","cqđt","tt","cq","ttlt","ttbyt","đkxt","tnhc","hsd","tvtu","ds","gđkt","ddvn","dtsd","cđdc","ttbdvhng","tthtcđ","tnxh","p","gcnđkkd","bk","gvth","ktnn","nhđtpt","ncc","bst","nb","tnnd","gcmnd","đhbkđn","kddv","bchcd","vqg","ubmttqvn","ktqd","tnxp","gtcc","hkdd","chcn","pcth","cph"}
        self.prefixAndSuffixList = {"de","dis","ex","il","im","in","mis","non","pre","pro","re","un","able","al","er","est","ful","ible","ily","ing","less","ly","ness","y"}

    def asciiTransform(self, inputString, maxLen):
        vector = []
        for i in range(max(len(inputString), maxLen)):
            if (i < len(inputString)):
                vector.append(ord(inputString[i]))
            else:
                vector.append(0)
        return vector

    def CountChar(self, nsw, test):
        count = 0
        for j in range(len(nsw)):
            charString = nsw[j]
            if (charString in test):
                count = count + 1
        return count

    def containsKeyword(self, myString, keywords):
        for keyword in keywords :
            if keyword in myString:
                return 1
            
        return 0

    def makeSingleCase(self, nsw,contextBefore,contextAfter):
        vector = []
# //		feature count 13
        vector.append(len(nsw))
        vector.append(self.CountChar(nsw, self.dauCharList))
        vector.append(self.CountChar(nsw, self.nguyenAmCharList))
        vector.append(self.CountChar(nsw, self.phuAmCharList))
        vector.append(self.CountChar(nsw, self.nuocNgoaiCharList))
        vector.append(self.CountChar(nsw, self.upperCharList))
        vector.append(self.CountChar(nsw, self.lowerCharList))
        vector.append(self.CountChar(nsw, self.numCharList))
        vector.append(self.CountChar(nsw, self.puncDateCharList))
        vector.append(self.CountChar(nsw, self.puncHourCharList))
        vector.append(self.CountChar(nsw, self.abbrevCharList))
        vector.append(self.CountChar(nsw, self.otherCharList))
        vector.append(self.CountChar(nsw, self.mathCharList))
# //		feature occur nsw 9
        vector.append(self.containsKeyword(nsw.lower(),self.measureUnitCharList))
        vector.append(self.containsKeyword(nsw.lower(),self.currencyCharList))
        vector.append(self.containsKeyword(nsw.lower(),self.measureCharList))
        vector.append(self.containsKeyword(nsw,self.romanCharList))
        vector.append(self.containsKeyword(nsw.lower(),self.hourTextList))
        vector.append(self.containsKeyword(nsw.lower(),self.minuteTextList))
        vector.append(self.containsKeyword(nsw.lower(),self.secTextList))
        vector.append(self.containsKeyword(nsw.lower(),self.abbrList))
        vector.append(self.containsKeyword(nsw.lower(),self.prefixAndSuffixList))
# //		feature occur context before 8
        vector.append(self.containsKeyword(contextBefore.lower(),self.dateTextList))
        vector.append(self.containsKeyword(contextBefore.lower(),self.monthTextList))
        vector.append(self.containsKeyword(contextBefore.lower(),self.quyTextList))
        vector.append(self.containsKeyword(contextBefore.lower(),self.yearTextList))
        vector.append(self.containsKeyword(contextBefore.lower(),self.timeTextList))
        vector.append(self.containsKeyword(contextBefore.lower(),self.fractionTextList))
        vector.append(self.containsKeyword(contextBefore.lower(),self.scoreTextList))
        vector.append(self.containsKeyword(contextBefore.lower(),self.romanTextList))
# //		ascii 20 nsw 30 context
        vector = vector + self.asciiTransform(nsw,30)
        vector = vector + self.asciiTransform(contextBefore,40)
        # print(len(self.asciiTransform(nsw,30)))
        vector = vector + self.asciiTransform(contextAfter,40)
        return vector

    def makeFeatureSentence(self,inputString):
        listVector = []
        pattern = "<nsw[^>]*>([^<]*)</nsw>"
        indexs = [(m.start(0), m.end(0), m.group(1)) for m in re.finditer(pattern, inputString)]
        lenString = len(inputString)
        for i in range(len(indexs)):
            start = indexs[i][0]
            end = indexs[i][1]
            nsw = indexs[i][2]
            if len(nsw) >= 30:
                raise NameError('Invalid Input')
            contextBefore = inputString[max(0, start - 40):start]
            contextAfter  = inputString[end:min(lenString, end + 40)]
            # print("contextBefore : " , len(contextBefore))
            # print("contextAfter  : " , len(contextAfter))
            # print("nsw           : " , nsw)
            listVector.append(self.makeSingleCase(nsw,contextBefore,contextAfter))
        return listVector

if __name__ == '__main__':

    vector = Vectorizator()
    # output = vector.makeFeatureSentence('Hiện nay ở Việt Nam rất nhiều cuốn sách đã được in có đề cập đến năm Can Chi , đặc biệt là các sách nghiên cứu về lịch sử , nhưng lại rất hay bị nhầm lẫn vì người viết có thể không biết cách tính Can Chi.Phương pháp tính can <nsw> chi0123456789TânCanhKỷMâụĐinhBínhẤtGiápQuýNhâm0123456789CanhTânNhâmQuýGiápẤtBínhĐinhMâụKỷ </nsw> ChiTý SưủDần MãoThìn TỵNgọ MùiThân DâụTuất HơịGiáp15141312111Bính13353433323Mâụ25155554535Canh37271775747Nhâm49392919959Giáp0424446484Bính1636567696Mâụ0828486888Canh0020406080Nhâm1232527292Ất0525456585Đinh1737577797Kỷ0929496989Tân0121416181Quý1333537393Bính0626466686Mâụ1838587898Canh1030507090Nhâm0222426282Giáp1434547494Đinh0727476787Kỷ1939597999Tân1131517191Quý0323436383Ất1535557595 .')
    # print(np.shape(output))
    data_csv = pd.read_csv("data/final.csv", error_bad_lines=False) 
    data_in_csv = data_csv['context']
    label_csv = data_csv['label']
    feat_data = defaultdict(list)
    for i in range(1,len(data_in_csv)):
        if not (i%1000):
            print(i)
        try:
        # print(data_in_csv[i])
            shape = np.shape(vector.makeFeatureSentence(data_in_csv[i]))
            if shape[1] != 140:
                print(data_in_csv[i])
                break
            feat_data[label_csv[i]] = feat_data[label_csv[i]] + (vector.makeFeatureSentence(data_in_csv[i]))
        except:
            print(" fail in : ",i)
            pass

    # print(feat_data)
    data_train = []
    label = []
    for key in feat_data.keys():
        for item in feat_data[key]:
            data_train.append(item)
            label.append(key)
    print(np.shape(data_train))
    with open('feature/feature_new.pkl', 'wb') as handle:
        pickle.dump(data_train, handle)
    with open('feature/label_new.pkl', 'wb') as handle:
        pickle.dump(label, handle)