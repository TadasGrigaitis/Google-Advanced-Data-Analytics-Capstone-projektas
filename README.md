# Google Advanced Data Analytics Capstone projektas
**Apžvalga**

Projekto tikslas - naudojant žmogiškųjų išteklių departamento duomenis, sudaryti regresijos modelį, pagal kurį būtų galima prognozuoti, ar darbuotojas paliks įmonę, ar ne. Naudojant python programavimo kalbą, sudariau logistinės regresijos modelį, kuris pasižymėjo šiomis prognozių statistikomis: _precision_ 79 %, _recall_ - 82 %, _f1 rodiklis_ - 80 %. 

**Aktualumas** 

Tiriamosios duomenų analyzės metu ir regresijos modelio pagalba gautomis įžvalgomis įmonė galėtų suprasti kodėl darbuotojai palieka darbo vietas.

**Duomenys**

Google Advanced Data Analytics autoriai pateikė Salifort Motors įmonės 10 metų laikotarpio žmogiškųjų išteklių departamento duomenis. Žemiau pateikiama viena iš projekto vizualizacijų -darbuotojų išėjimo iš ir pasilikimo įmonėje histograma, išskirstyta pagal departamentą.
<p align="center">
<img src ="https://github.com/user-attachments/assets/3827b8bf-b053-44d9-8bee-56d2dbbbe8e4"  width="400" height="400">
</p>

**Modelis** 

Modelio rezultatams vizualiai įvertinti naudojau painiavos matricą (angl. confusion matrix). Ji pavaizduota žemiau:
<p align="center">
<img src ="https://github.com/user-attachments/assets/1c70dec8-b8c2-417e-917f-00bd63c878f7" width="300" height="300">
</p>

Statistikos indikuoja visai neblogus rezultatus, tačiau jei svarbiausia prognozuoti iš darbo išeinančius darbuotojus, tuomet įverčiai yra gerokai mažesni. Žemiau galite matyti regresijos modelio statistikas:
<p align="center">
<img src ="https://github.com/user-attachments/assets/282f788b-5c01-40ef-aecf-bc722a741a32" width="350" height="150">
</p>

**Išvada**

Projektas atskleidžia praktinį duomenų analizės pritaikymą, siekiant spręsti aktualią personalo valdymo problemą – darbuotojų išlaikymą. Logistinės regresijos modelis su pasiektais tikslumo rodikliais (_precision_ 79 %, _recall_ 82 %, _f1 rodiklis_ 80 %) rodo pakankamai gerą prognozių kokybę ir gali būti vertingas įrankis verslo sprendimuose.
