$ontext
DICE2023-beta-4-3-10 as of October 16, 2023
$offtext

$title October 12, 2023 (DICE2023-b-4-3-10.gms)

set        t  Time periods (5 years per period)                   /1*81/
PARAMETERS
** If optimal control
        ifopt    Indicator where optimized is 1 and base is 0     /1/
** Population and technology
        gama     Capital elasticity in production function        /.300    /
        pop1     Initial world population 2020 (millions)         /7752.9  /
        popadj   Growth rate to calibrate to 2050 pop projection  /0.145   /
        popasym  Asymptotic population (millions)                 /10825.  /
        dk       Depreciation rate on capital (per year)          /0.100    /
        q1       Initial world output 2020 (trill 2019 USD)       /135.7   /
        AL1       Initial level of total factor productivity       /5.84 /
        gA1      Initial growth rate for TFP per 5 years          /0.066   /
        delA     Decline rate of TFP per 5 years                  /0.0015  /        
** Emissions parameters and Non-CO2 GHG with sigma = emissions/output
        gsigma1   Initial growth of sigma (per year)                   / -0.015 /
        delgsig   Decline rate of gsigma per period                    /.96/
        asymgsig   Asympototic gsigma                                  /-.005/
        e1        Industrial emissions 2020 (GtCO2 per year)           / 37.56  /
        miu1      Emissions control rate historical 2020               / .05    /
        fosslim   Maximum cumulative extraction fossil fuels (GtC)     / 6000   /
        CumEmiss0 Cumulative emissions 2020 (GtC)                      / 633.5/
* Climate damage parameters
        a1        Damage intercept                                    /0      /
        a2base    Damage quadratic term rev 01-13-23                  /0.003467/
        a3        Damage exponent                                     /2.00   /
** Abatement cost
        expcost2  Exponent of control cost function                   / 2.6  /
        pback2050 Cost of backstop 2019$ per tCO2 2050                / 515.  /
        gback     Initial cost decline backstop cost per year         / -.012 /
        cprice1   Carbon price 2020 2019$ per tCO2                    / 6    /
        gcprice   Growth rate of base carbon price per year           /.025   /
** Limits on emissions controls
        limmiu2070    Emission control limit from 2070          /1.0/
        limmiu2120    Emission control limit from 2120          /1.1/
        limmiu2200    Emission control limit from 2220          /1.05/
        limmiu2300    Emission control limit from 2300          /1.0/      
        delmiumax     Emission control delta limit per period   / 0.12/
** Preferences, growth uncertainty, and timing
        betaclim  Climate beta                                      / 0.5  /
        elasmu    Elasticity of marginal utility of consumption     / 1.50 /
        prstp     Pure rate of social time preference               /.001/
        pi        Capital risk premium                              / .05  /
        rartp       Risk-adjusted rate of time preference
        k0        Initial capital stock calibrated (1012 2019 USD)  / 295  /
        siggc1    Annual standard deviation of consumption growth   / .01 /
        sig1      Carbon intensity 2020 kgCO2-output 2020
** Scaling so that MU(C(1)) = 1 and objective function = PV consumption
        tstep       Years per Period                               / 5  /
        SRF         Scaling factor discounting                     /1000000/
        scale1      Multiplicative scaling coefficient             / 0.00891061 /
        scale2      Additive scaling coefficient                   /-6275.91/ ;
** Program control variables
sets     tfirst(t), tlast(t), tearly(t), tlate(t);
PARAMETERS
        L(t)           Level of population and labor
        aL(t)          Level of total factor productivity
        sigma(t)       CO2-emissions output ratio
        sigmatot(t)    GHG-output ratio
        gA(t)          Growth rate of productivity from
        gL(t)          Growth rate of labor and population
        gcost1         Growth of cost factor
        gsig(t)        Change in sigma (rate of decarbonization)
        eland(t)       Emissions from deforestation (GtCO2 per year)
        cost1tot(T)    Abatement cost adjusted for backstop and sigma
        pbacktime(t)   Backstop price 2019$ per ton CO2
        optlrsav       Optimal long-run savings rate used for transversality
        scc(t)         Social cost of carbon
        cpricebase(t)  Carbon price in base case
        ppm(t)         Atmospheric concentrations parts per million
        atfrac2020(t)  Atmospheric share since 2020
        atfrac1765(t)  Atmospheric fraction of emissions since 1765
        abaterat(t)    Abatement cost per net output
        miuup(t)       Upper bound on miu
        gbacktime(t)   Decline rate of backstop price
** Precautionary dynamic parameters
        varpcc(t)         Variance of per capita consumption 
        rprecaut(t)       Precautionary rate of return
        RR(t)             STP with precautionary factor
        RR1(t)            STP factor without precautionary factor;
** Time preference for climate investments and precautionary effect
        rartp           = exp( prstp + betaclim*pi)-1;
        varpcc(t)       =  min(Siggc1**2*5*(t.val-1),Siggc1**2*5*47);       
        rprecaut(t)     = -0.5*varpcc(t)*elasmu**2;
        RR1(t)          = 1/((1+rartp)**(tstep*(t.val-1)));
        RR(t)           = RR1(t)*(1+rprecaut(t))**(-tstep*(t.val-1));
        L("1") = pop1; loop(t, L(t+1)=L(t);); loop(t, L(t+1)  = L(t)*(popasym/L(t))**popadj ;);
        gA(t)           = gA1*exp(-delA*5*((t.val-1))); aL("1") = AL1; loop(t, aL(t+1)=aL(t)/((1-gA(t))););
        optlrsav        =(dk + .004)/(dk + .004*elasmu + rartp)*gama;
        cpricebase(t)   = cprice1*(1+gcprice)**(5*(t.val-1));
        pbacktime(t)    = pback2050*exp(-5*(.01)*(t.val-7));
        pbacktime(t)$(t.val > 7) = pback2050*exp(-5*(.001)*(t.val-7));
        sig1            = e1/(q1*(1-miu1)); sigma("1") = sig1;
        gsig(t)         = min(gsigma1*delgsig **((t.val-1)),asymgsig);
        loop(t, sigma(t+1)  = sigma(t)*exp(5*gsig(t)););
** Emissions limits
        miuup('1')= .05; miuup('2')= .10; miuup(t)$(t.val > 2)  = (delmiumax*(t.val-1));
        miuup(t)$(t.val > 8)  = 0.85+.05*(t.val-8); miuup(t)$(t.val > 11) = limmiu2070;
        miuup(t)$(t.val > 20) = limmiu2120; miuup(t)$(t.val > 37) = limmiu2200; miuup(t)$(t.val > 57) = limmiu2300;       
** Include file for non-CO2 GHGs
* nonco2 Parameters      
Parameters
        CO2E_GHGabateB(t)         Abateable non-CO2 GHG emissions base
        CO2E_GHGabateact(t)       Abateable non-CO2 GHG emissions base (actual)
        F_Misc(t)                 Non-abateable forcings (GHG and other)
        emissrat(t)               Ratio of CO2e to industrial emissions
        sigmatot(t)               Emissions output ratio for CO2e       
        FORC_CO2(t)               CO2 Forcings
;      
** Parameters for non-industrial emission
** Assumes abateable share of non-CO2 GHG is 65%
Parameters
        eland0         Carbon emissions from land 2015 (GtCO2 per year)  / 5.9    /
        deland         Decline rate of land emissions (per period)       / .1     /      
        F_Misc2020     Non-abatable forcings 2020                       /  -0.054    /
        F_Misc2100     Non-abatable forcings 2100                        / .265/       
        F_GHGabate2020 Forcings of abatable nonCO2 GHG                   / 0.518 /
        F_GHGabate2100 Forcings of abatable nonCO2 GHG                   / 0.957 /

        ECO2eGHGB2020  Emis of abatable nonCO2 GHG GtCO2e  2020             /  9.96/     
        ECO2eGHGB2100  Emis of abatable nonCO2 GHG GtCO2e  2100             /  15.5 /     
        emissrat2020   Ratio of CO2e to industrial CO2 2020                 / 1.40 /
        emissrat2100   Ratio of CO2e to industrial CO2 2020                 / 1.21 /       
        Fcoef1         Coefficient of nonco2 abateable emissions            /0.00955/
        Fcoef2         Coefficient of nonco2 abateable emissions            /.861/
        ;
** Parameters emissions and non-CO2 
        eland(t) = eland0*(1-deland)**(t.val-1); eland(t) = eland0*(1-deland)**(t.val-1);       
        CO2E_GHGabateB(t)=ECO2eGHGB2020+((ECO2eGHGB2100-ECO2eGHGB2020)/16)*(t.val-1)$(t.val le 16)+((ECO2eGHGB2100-ECO2eGHGB2020))$(t.val ge 17);
        F_Misc(t)=F_Misc2020 +((F_Misc2100-F_Misc2020)/16)*(t.val-1)$(t.val le 16)+((F_Misc2100-F_Misc2020))$(t.val ge 17);
        emissrat(t) = emissrat2020 +((emissrat2100-emissrat2020)/16)*(t.val-1)$(t.val le 16)+((emissrat2100-emissrat2020))$(t.val ge 17); 
        sigmatot(t) = sigma(t)*emissrat(t);
        cost1tot(t) = pbacktime(T)*sigmatot(T)/expcost2/1000;
VARIABLES
        ECO2(t)         Total CO2 emissions (GtCO2 per year)
        ECO2E(t)        Total CO2e emissions including abateable nonCO2 GHG (GtCO2 per year)
        EIND(t)         Industrial CO2 emissions (GtCO2 per yr)
        F_GHGabate      Forcings abateable nonCO2 GHG     
;
Equations
        ECO2eq(t)         CO2 Emissions equation
        ECO2Eeq(t)        CO2E Emissions equation
        EINDeq(t)        Industrial CO2 equation
        F_GHGabateEQ(t)
;        
* Program control definitions
        tfirst(t) = yes$(t.val eq 1);
        tlast(t)  = yes$(t.val eq card(t));
VARIABLES
        MIU(t)          Emission control rate GHGs
        C(t)            Consumption (trillions 2019 US dollars per year)
        K(t)            Capital stock (trillions 2019 US dollars)
        CPC(t)          Per capita consumption (thousands 2019 USD per year)
        I(t)            Investment (trillions 2019 USD per year)
        S(t)            Gross savings rate as fraction of gross world product
        Y(t)            Gross world product net of abatement and damages (trillions 2019 USD per year)
        YGROSS(t)       Gross world product GROSS of abatement and damages (trillions 2019 USD per year)
        YNET(t)         Output net of damages equation (trillions 2019 USD per year)
        DAMAGES(t)      Damages (trillions 2019 USD per year)
        DAMFRAC(t)      Damages as fraction of gross output
        ABATECOST(t)    Cost of emissions reductions  (trillions 2019 USD per year)
        MCABATE(t)      Marginal cost of abatement (2019$ per ton CO2)
        CCATOT(t)       Total carbon emissions (GtC)
        PERIODU(t)      One period utility function
        CPRICE(t)       Carbon price (2019$ per ton of CO2)
        TOTPERIODU(t)   Period utility
        UTILITY         Welfare function
        RFACTLONG(t)
        RSHORT(t)       Real interest rate with precautionary(per annum year on year)
        RLONG(t)        Real interest rate from year 0 to T
;
NONNEGATIVE VARIABLES  MIU, TATM, MAT, MU, ML, Y, YNET, YGROSS, C, K, I, RFACTLONG;
EQUATIONS
*Emissions and Damages
        CCATOTEQ(t)      Cumulative total carbon emissions
        DAMFRACEQ(t)     Equation for damage fraction
        DAMEQ(t)         Damage equation
        ABATEEQ(t)       Cost of emissions reductions equation
        MCABATEEQ(t)     Equation for MC abatement
        CARBPRICEEQ(t)   Carbon price equation from abatement
*Economic variables
        YGROSSEQ(t)      Output gross equation
        YNETEQ(t)        Output net of damages equation
        YY(t)            Output net equation
        CC(t)            Consumption equation
        CPCE(t)          Per capita consumption definition
        SEQ(t)           Savings rate equation
        KK(t)            Capital balance equation
        RSHORTEQ(t)      Short-run interest rate equation
        RLONGeq(t)       Long-run interest rate equation
        RFACTLONGeq(t)   Long interest factor
* Utility
        TOTPERIODUEQ(t)  Period utility
        PERIODUEQ(t)     Instantaneous utility function equation
        UTILEQ           Objective function      ;

** Include file for DFAIR model and climate equations
** Equals old FAIR with recalibrated parameters for revised F2xco2 and Millar model.
** Deletes nonnegative reservoirs. See explanation below


sets     tfirst(t), tlast(t);

PARAMETERS
         yr0     Calendar year that corresponds to model year zero         /2020/
        emshare0 Carbon emissions share into Reservoir 0   /0.2173/
        emshare1 Carbon emissions share into Reservoir 1    /0.224/
        emshare2 Carbon emissions share into Reservoir 2    /0.2824/
        emshare3 Carbon emissions share into Reservoir 3    /0.2763/
        tau0    Decay time constant for R0  (year)                            /1000000/
        tau1    Decay time constant for R1  (year)                            /394.4/
        tau2    Decay time constant for R2  (year)       /36.53/
        tau3    Decay time constant for R3  (year) /4.304/

        teq1    Thermal equilibration parameter for box 1 (m^2 per KW)         /0.324/
        teq2    Thermal equilibration parameter for box 2 (m^2 per KW)        /0.44/
        d1      Thermal response timescale for deep ocean (year)               /236/
        d2      Thermal response timescale for upper ocean (year)              /4.07/
 
        irf0    Pre-industrial IRF100 (year)                                        /32.4/
        irC      Increase in IRF100 with cumulative carbon uptake (years per GtC)    /0.019/
        irT      Increase in IRF100 with warming (years per degree K)                /4.165/
        fco22x   Forcings of equilibrium CO2 doubling (Wm-2)                        /3.93/       

** INITIAL CONDITIONS TO BE CALIBRATED TO HISTORY
** CALIBRATION
       mat0   Initial concentration in atmosphere in 2020 (GtC)       /886.5128014/

       res00  Initial concentration in Reservoir 0 in 2020 (GtC)      /150.093 /
       res10  Initial concentration in Reservior 1 in 2020 (GtC)      /102.698 /
       res20  Initial concentration in Reservoir 2 in 2020 (GtC)      /39.534  /
       res30  Initial concentration in Reservoir 3 in 2020 (GtC)      / 6.1865 /



       mateq      Equilibrium concentration atmosphere  (GtC)            /588   /
       tbox10    Initial temperature box 1 change in 2020 (C from 1765)  /0.1477  /
       tbox20    Initial temperature box 2 change in 2020 (C from 1765)  /1.099454/
       tatm0     Initial atmospheric temperature change in 2020          /1.24715 /     
        
;
VARIABLES
*Note: Stock variables correspond to levels at the END of the period
        FORC(t)        Increase in radiative forcing (watts per m2 from 1765)
        TATM(t)        Increase temperature of atmosphere (degrees C from 1765)     
        TBOX1(t)       Increase temperature of box 1 (degrees C from 1765)
        TBOX2(t)       Increase temperature of box 2 (degrees C from 1765)
        RES0(t)        Carbon concentration in Reservoir 0 (GtC from 1765)
        RES1(t)        Carbon concentration in Reservoir 1 (GtC from 1765)
        RES2(t)        Carbon concentration in Reservoir 2 (GtC from 1765)
        RES3(t)        Carbon concentration in Reservoir 3 (GtC from 1765)
        MAT(t)         Carbon concentration increase in atmosphere (GtC from 1765)
        CACC(t)        Accumulated carbon in ocean and other sinks (GtC)
        IRFt(t)        IRF100 at time t
        alpha(t)       Carbon decay time scaling factor
        SumAlpha      Placeholder variable for objective function;

**** IMPORTANT PROGRAMMING NOTE. Earlier implementations has reservoirs as non-negative.
**** However, these are not physical but mathematical solutions.
**** So, they need to be unconstrained so that can have negative emissions.
NONNEGATIVE VARIABLES   TATM, MAT,  IRFt, alpha

EQUATIONS
        FORCE(t)        Radiative forcing equation
        RES0LOM(t)      Reservoir 0 law of motion
        RES1LOM(t)      Reservoir 1 law of motion
        RES2LOM(t)      Reservoir 2 law of motion
        RES3LOM(t)      Reservoir 3 law of motion
        MMAT(t)         Atmospheric concentration equation
        Cacceq(t)       Accumulated carbon in sinks equation
        TATMEQ(t)       Temperature-climate equation for atmosphere
        TBOX1EQ(t)      Temperature box 1 law of motion
        TBOX2EQ(t)      Temperature box 2 law of motion
        IRFeqLHS(t)     Left-hand side of IRF100 equation
        IRFeqRHS(t)     Right-hand side of IRF100 equation
;
** Equations of the model
    res0lom(t+1)..   RES0(t+1) =E=  (emshare0*tau0*alpha(t+1)*(Eco2(t+1)/3.667))*(1-exp(-tstep/(tau0*alpha(t+1))))+Res0(t)*exp(-tstep/(tau0*alpha(t+1)));
    res1lom(t+1)..   RES1(t+1) =E=  (emshare1*tau1*alpha(t+1)*(Eco2(t+1)/3.667))*(1-exp(-tstep/(tau1*alpha(t+1))))+Res1(t)*exp(-tstep/(tau1*alpha(t+1)));
    res2lom(t+1)..   RES2(t+1) =E=  (emshare2*tau2*alpha(t+1)*(Eco2(t+1)/3.667))*(1-exp(-tstep/(tau2*alpha(t+1))))+Res2(t)*exp(-tstep/(tau2*alpha(t+1)));
    res3lom(t+1)..   RES3(t+1) =E=  (emshare3*tau3*alpha(t+1)*(Eco2(t+1)/3.667))*(1-exp(-tstep/(tau3*alpha(t+1))))+Res3(t)*exp(-tstep/(tau3*alpha(t+1)));
    mmat(t+1)..      MAT(t+1)  =E=   mateq+Res0(t+1)+Res1(t+1)+Res2(t+1)+Res3(t+1);
    cacceq(t)..      Cacc(t)   =E=  (CCATOT(t)-(MAT(t)-mateq));
    force(t)..       FORC(t)    =E=  fco22x*((log((MAT(t)/mateq))/log(2))) + F_Misc(t)+F_GHGabate(t);
    
    tbox1eq(t+1)..   Tbox1(t+1) =E=  Tbox1(t)*exp(-tstep/d1)+teq1*Forc(t+1)*(1-exp(-tstep/d1));
    tbox2eq(t+1)..   Tbox2(t+1) =E=  Tbox2(t)*exp(-tstep/d2)+teq2*Forc(t+1)*(1-exp(-tstep/d2));
    tatmeq(t+1)..    TATM(t+1)  =E=   Tbox1(t+1)+Tbox2(t+1);
    irfeqlhs(t)..    IRFt(t)   =E=  ((alpha(t)*emshare0*tau0*(1-exp(-100/(alpha(t)*tau0))))+(alpha(t)*emshare1*tau1*(1-exp(-100/(alpha(t)*tau1))))+(alpha(t)*emshare2*tau2*(1-exp(-100/(alpha(t)*tau2))))+(alpha(t)*emshare3*tau3*(1-exp(-100/(alpha(t)*tau3)))));
    irfeqrhs(t)..    IRFt(t)   =E=  irf0+irC*Cacc(t)+irT*TATM(t);

**  Upper and lower bounds for stability
MAT.LO(t)       = 10;
TATM.UP(t)      = 20;
TATM.lo(t)      = .5;
alpha.up(t) = 100;
alpha.lo(t) = 0.1;

* Initial conditions
MAT.FX(tfirst)    = mat0;
TATM.FX(tfirst)   = tatm0;
Res0.fx(tfirst) = Res00;
Res1.fx(tfirst) = Res10;
Res2.fx(tfirst) = Res20;
Res3.fx(tfirst) = Res30;
Tbox1.fx(tfirst) = Tbox10;
Tbox2.fx(tfirst) = Tbox20;

** Solution options
option iterlim = 99900;
option reslim = 99999;
option solprint = off;
option limrow = 0;
option limcol = 0;
 

**** Equations of the model
**Emissions and Damages
 eco2eq(t)..          ECO2(t)        =E= (sigma(t)*YGROSS(t) + eland(t))*(1-(MIU(t)));
 eindeq(t)..          EIND(t)        =E= (sigma(t)*YGROSS(t))*(1-(MIU(t)));
 eco2Eeq(t)..         ECO2E(t)       =E= (sigma(t)*YGROSS(t) + eland(t) + CO2E_GHGabateB(t))*(1-(MIU(t)));
 F_GHGabateEQ(t+1)..  F_GHGabate(t+1) =E= Fcoef2*F_GHGabate(t)+ Fcoef1*CO2E_GHGabateB(t)*(1-(MIU(t)));
 ccatoteq(t+1)..      CCATOT(t+1)    =E= CCATOT(t) +  ECO2(T)*(5/3.666) ;
 damfraceq(t) ..      DAMFRAC(t)     =E= (a1*(TATM(t)))+(a2base*(TATM(t))**a3) ;
 dameq(t)..           DAMAGES(t)     =E= YGROSS(t) * DAMFRAC(t);
 abateeq(T)..         ABATECOST(T)   =E= YGROSS(T) * COST1TOT(T) * (MIU(T)**EXPCOST2);
 mcabateeq(t)..       MCABATE(t)     =E= pbacktime(t) * MIU(t)**(expcost2-1);
 carbpriceeq(t)..     CPRICE(t)      =E= pbacktime(t) * (MIU(t))**(expcost2-1);
**Economic variables
 ygrosseq(t)..        YGROSS(t)      =E= (AL(t)*(L(t)/1000)**(1-gama))*(K(t)**gama);
 yneteq(t)..          YNET(t)        =E= YGROSS(t)*(1-damfrac(t));
 yy(t)..              Y(t)           =E= YNET(t) - ABATECOST(t);
 cc(t)..              C(t)           =E= Y(t) - I(t);
 cpce(t)..            CPC(t)         =E= 1000 * C(t) / L(t);
 seq(t)..             I(t)           =E= S(t) * Y(t);
 kk(t+1)..            K(t+1)         =L= (1-dk)**tstep * K(t) + tstep * I(t);
 RFACTLONGeq(t+1)..   RFACTLONG(t+1) =E= SRF*(cpc(t+1)/cpc('1'))**(-elasmu)*rr(t+1);
 RLONGeq(t+1)..       RLONG(t+1)     =E= -log(RFACTLONG(t+1)/SRF)/(5*t.val);
 RSHORTeq(t+1)..      RSHORT(t+1)    =E= -log(RFACTLONG(t+1)/Rfactlong(t))/5;
** Welfare functions 
 periodueq(t)..       PERIODU(t)     =E= ((C(T)*1000/L(T))**(1-elasmu)-1)/(1-elasmu)-1;
 totperiodueq(t)..    TOTPERIODU(t)  =E= PERIODU(t) * L(t) * RR(t);
 utileq..             UTILITY        =E= tstep * scale1 * sum(t,  TOTPERIODU(t)) + scale2;

* Various control rate limits, initial and terminal conditions
miu.up(t)       = miuup(t);
K.LO(t)         = 1;
C.LO(t)         = 2;
CPC.LO(t)       = .01;
RFACTLONG.lo(t) =.0001;
*set lag10(t) ;
*lag10(t)                =  yes$(t.val gt card(t)-10);
*S.FX(lag10(t))          = optlrsav;
s.fx(t)$(t.val > 37)    =.28;
ccatot.fx(tfirst)       = CumEmiss0;
K.FX(tfirst)            = k0;
F_GHGabate.fx(tfirst)   = F_GHGabate2020;
RFACTLONG.fx(tfirst)    = 1000000; 

** Solution options
option iterlim = 99900; option reslim = 99999; option solprint = off; option limrow = 0; option limcol = 0;

** Initial solution
model  CO2 /all/;
ifopt=1;
solve CO2 maximizing UTILITY using nlp ;
 
**** STATEMENTS FOR DEFINITIONS AND PUT STATEMENTS FOR MAJOR SCENARIOS
* Solve equations for optimal case

ifopt=1;

* Solve
solve CO2 maximizing UTILITY using nlp ;
solve CO2 maximizing UTILITY using nlp ;
solve CO2 maximizing UTILITY using nlp ;

*Post-Solution Parameter-Assignment

scc(t) = -1000*eco2eq.m(t)/(.00001+cc.m(t));
ppm(t)    = mat.l(t)/2.13;
abaterat(t) = abatecost.l(t)/y.l(t);
atfrac2020(t) = ((mat.l(t)-mat0)/(ccatot.l(t)+.00001-CumEmiss0  ));
atfrac1765(t) = ((mat.l(t)-mateq)/(.00001+ccatot.l(t)  ));
FORC_CO2(t) = fco22x*((log((MAT.l(t)/mateq))/log(2))); 
* ---- Output for the eta-letter cross-check (added; model unchanged) ----
scalar peakT, peakyear, scc2025, scc2050;
peakT = smax(t, tatm.l(t));
peakyear = smax(t$(tatm.l(t) = peakT), 2015 + 5*t.val);
scc2025 = scc('2'); scc2050 = scc('7');
option decimals = 8;
display elasmu, CO2.modelstat, CO2.solvestat, utility.l, scc2025, scc2050, peakT, peakyear;
display scc, tatm.l, miu.l, s.l, cpc.l;
