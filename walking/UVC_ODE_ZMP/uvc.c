/*
Apply UVC to KHR.  May 8,2022 : Vre 1.0
*/

#include "stdio.h"
#include "kcb5.h"
#include "i2c.h"
#include "uart.h"
#include "pio.h"
#include "rom.h"
#include "timer.h"
#include "ics.h"
#include "ad.h"
#include "dac.h"
#include "math.h"

#define OFS_L 129.0			//初期の股関節までの距離（膝カク防止）
#define OFS_S 213			//初期の股関節サーボ角
#define CHG_SVA 1718.9		//サーボ角に変換
#define HEIGHT 185			//185脚長
#define RET_F_LEG 3			//遊脚追従率
#define RST_F_LEG 3			//軸足垂直戻率

//// グローバル変数
int16_t K0W[2];			//股関節前後方向書込用
int16_t K1W[2];			//股関節横方向書込用
int16_t K2W[2];			//股関節横方向書込用
int16_t HW [2];			//膝関節書込用
int16_t A0W[2];			//足首上下方向書込用
int16_t A1W[2];			//足首横方向書込用
int16_t U0W[2];			//肩前後方向書込用
int16_t U1W[2];			//肩横後方向書込用
int16_t U2W[2];			//肩ヨー向書込用
int16_t EW[2];			//肘書込用
int16_t WESTW;			//腰回転書込用
int16_t HEADW;			//頭回転書込用
int16_t K0R,K0RB,U0R,U0L,U1R,U1L,EWS;
int16_t K0L,K0LB,U0WB;
int16_t K0WB;
int16_t jikuasi;
int16_t motCt,motCtBak,motCtBak2,motCtdat;
int16_t mode,modeNxt,subMode,keyMode;
int16_t	pitch_gyr,roll_gyr,yaw_gyr;
int16_t	cycle,tst0;
int16_t	walkCt,walkCtLim;		//歩数
int16_t	p_ofs,r_ofs;
int16_t ir,ip,ira,ipa;
int16_t irb,ipb,ct;
int16_t	pitchs,rolls,pitch_ofs, roll_ofs, yaw, yaw_ofs;
int16_t	landF,landB;

int32_t	tBak, pitchi;
uint32_t tNow;

uint8_t cmd[2];
uint8_t ff[45];
uint8_t dsp[110];
uint8_t krr[4];
int8_t	kn;
int8_t	LEDct;	//LED点灯カウンタ

float fwctEnd,fwct,fwctUp;
float pitch,roll,pitcht,rollt;
float pitch_gyrg,         roll_gyrg;
float wk,wt;
float dyi,dyib,dyis;
float dxi,dxib,dxis;
float rollg,fw,fwi,fws,sw,freeBak,tt0;
float supportingLeg,swingLeg;	//支持脚、遊脚股関節振出角度
float footH;					//脚上げ高さ
float swx,swy,swMax;			//横振り巾
float autoH,fh,fhMax;			//脚上げ高さ

//// 汎用 ////
int32_t	ii;
int16_t	i,j;
float k,k0,k1,ks,kxy,kl;



////////////////////////
//// ターミナル表示 ////
////////////////////////
void printS(char *StringData){
	unsigned char count = 0;
	while(StringData[count] != '\0')count++;
	uart_tx(UART_COM, (unsigned char*)StringData, 0, count);
}



////////////////////////
//// 遅延時間の設定 ////
////////////////////////
void delay(int32_t t)
{
	int32_t tb=0;
	t*=1000;
	timer_write(TIMER,t);
	timer_start(TIMER);
	t=0;
	while(t >= tb){
		tb=t;
		t=timer_read(TIMER);
	}
}



/////////////////////
//// I2Cアクセス ////
/////////////////////
uint8_t read8(uint8_t reg )
{
	cmd[0]=reg;
	//アドレスを0x28から左に1bitシフトし0x50、KONDOのI2Cライブラリバグ対応
	i2c_read(0x50, cmd, 1, ff, 1);  //BNO055_CHIP_ID_ADDR(0x28)
	return ff[0];
}
bool readLen(uint8_t reg, uint8_t len)
{
	cmd[0]=reg;
	//アドレスを0x28から左に1bitシフトし0x50、KONDOのI2Cライブラリバグ対応
	i2c_read(0x50, cmd, 1, ff, len);  //BNO055_CHIP_ID_ADDR(0x28)
	return true;
}
bool write8(uint8_t reg, uint8_t dat)
{
	cmd[0]=dat;
	//アドレスを0x28から左に1bitシフトし0x50、KONDOのI2Cライブラリバグ対応
	i2c_write(0x50, reg, cmd, 1);  //BNO055_CHIP_ID_ADDR(0x28)
	return true;
}



/////////////////////////////////////
//// 目標値まで徐々に間接を動かす ////
/////////////////////////////////////
void movSv(short *s,int d){
//引数 s:現サーボ位置 d:目標サーボ位置
	if(motCt<1)	*s = d;
	else		*s += (d-*s)/motCt;
}



///////////////////////////
//// 検出角度を校正する ////
///////////////////////////
void angAdj(void){

// **** pitch & roll ****
	if( pitchs<=ipb+1 && pitchs>=ipb-1 &&
		rolls <=irb+1 && rolls >=irb-1 	){
		++ip;
		ipa+=pitchs;
		ira+=rolls;
	}
	else {
		ip=0;
		ipa=0;
		ira=0;
	}
	ipb=pitchs;
	irb=rolls;
	sprintf( (char *)dsp,"P:%4d R:%4d C:%4d\r",pitchs,rolls,ip );
	printS((char *)dsp);
}



///////////////////////
//// 全方向転倒検知 ////
///////////////////////
void detAng(void){
	if( 0.35>fabs(pitch) && 0.35>fabs(roll) )return;
	sprintf( (char *)dsp," PRA:%4d %4d PRG:%4d %4d\r\n",pitchs,rolls,pitch_gyr,roll_gyr );
	printS((char *)dsp);

	ics_set_pos	( UART_SIO2, 1, 4735 +1350 );	//U0R
	ics_set_pos	( UART_SIO4, 1, 9320 -1350 );	//U0L
	ics_set_pos	( UART_SIO2, 4, 4800 +2700 );	//ER
	ics_set_pos	( UART_SIO4, 4,10150 -2700 );	//EL
	ics_set_pos	( UART_SIO1, 7, 7500 -2100 );	//K0R
	ics_set_pos	( UART_SIO3, 7, 7500 +2100 );	//K0L
	ics_set_pos ( UART_SIO1, 8, 9260 -4200 );	//HR
	ics_set_pos	( UART_SIO3, 8, 5740 +4200 );	//HL
	ics_set_pos	( UART_SIO1, 9, 7910 +2100 );	//A0R
	ics_set_pos	( UART_SIO3, 9, 7100 -2100 );	//A0L
	delay(30);
	ics_set_pos		( UART_SIO1, 5, 0 );	//K2R
	ics_set_pos		( UART_SIO2, 1, 0 );	//U0R
	ics_set_pos		( UART_SIO3, 5, 0 );	//K2L
	ics_set_pos		( UART_SIO4, 1, 0 );	//U0L
	ics_set_pos		( UART_SIO1, 6, 0 );	//K1R
	ics_set_pos		( UART_SIO2, 2, 0 );	//U1R
	ics_set_pos		( UART_SIO3, 6, 0 );	//K1L
	ics_set_pos		( UART_SIO4, 2, 0 );	//U1L
	ics_set_pos		( UART_SIO1, 7, 0 );	//K0R
	ics_set_pos		( UART_SIO3, 7, 0 );	//K0L
	ics_set_pos		( UART_SIO1, 8, 0 );	//HR
	ics_set_pos		( UART_SIO2, 4, 0 );	//ER
	ics_set_pos		( UART_SIO3, 8, 0 );	//HL
	ics_set_pos		( UART_SIO4, 4, 0 );	//EL
	ics_set_pos		( UART_SIO1, 9, 0 );	//A0R
	ics_set_pos		( UART_SIO3, 9, 0 );	//A0L
	sprintf( (char *)dsp,"turn over :%4d %4d\r\n",mode,pitchs );
	printS ( (char *)dsp );
	while(1){}
}



/////////////////////
//// UVC補助制御 ////
/////////////////////
void uvcSub(void){

	// ************ UVC終了時支持脚を垂直に戻す ************
	if( fwct<=landF ){

		// **** 左右方向 ****
		k = dyi/(11-fwct);
		dyi -= k;
		dyis += k;

		// **** 前後方向 ****
		if( dxi>RST_F_LEG ){
			dxi  -= RST_F_LEG;
			dxis -= RST_F_LEG;
		}
		else if( dxi<-RST_F_LEG ){
			dxi  += RST_F_LEG;
			dxis += RST_F_LEG;
		}
		else{
			dxis -= dxi;
			dxi   = 0;
		}
	}
	if(dyis> 70)	dyis=  70;
	if(dxis<-70)	dxis= -70;
	if(dxis> 70)	dxis=  70;

	// ************ 脚長制御 ************
	if(HEIGHT>autoH){						//脚長を徐々に復帰させる
		autoH +=(HEIGHT-autoH)*0.07;//0.07
	}
	else	autoH = HEIGHT;

	if( fwct>fwctEnd-landB && rollt>0){	//着地時姿勢を落し衝撃吸収
		autoH -= ( fabs(dyi-dyib)+fabs(dxi-dxib) ) * 0.02;
	}
	if(140>autoH)autoH=140;					//最低脚長補償
}

void uvcSub2(void){
	float k0,k1;

	// ************ UVC終了時支持脚を垂直に戻す ************

	// **** 左右方向 ****
	k1 = dyi/(fwctEnd-fwct+1);
	dyi -= k1;

	// **** 前後方向 ****
	k0 = dxi/(fwctEnd-fwct+1);
	dxi  -= k0;

	// **** 腰回転 ****
	wk -= wk/(fwctEnd-fwct+1);
	WESTW  -= WESTW/(fwctEnd-fwct+1);
	K2W[0]  = WESTW;
	K2W[1]  =-WESTW;

	if( fwct<=landF ){
		// **** 左右方向 ****
		dyis += k1;

		// **** 前後方向 ****
		dxis -= k0;
	}
	else{
		dyis -= dyis/(fwctEnd-fwct+1);
		dxis -= dxis/(fwctEnd-fwct+1);
	}

	// **** 脚長制御 ****
	autoH += (HEIGHT-autoH)/(fwctEnd-fwct+1);

	if(dyis> 70)	dyis=  70;
	if(dxis<-70)	dxis= -70;
	if(dxis> 70)	dxis=  70;
}



/////////////////
//// UVC制御 ////
/////////////////
void uvc(void){
	float pb,rb,k;

	// ************ 傾斜角へのオフセット適用 ************
	rb=roll;		//一時退避
	pb=pitch;
	k=sqrt(pitch*pitch+roll*roll);	//合成傾斜角
	if( k>0.033 ){
		k=(k-0.033)/k;
		pitch *=k;
		roll  *=k;
	}
	else{
		pitch =0;
		roll  =0;
	}


	// ************ 傾斜角に係数適用 ************
	rollt =0.25*roll;
	if(jikuasi==0)	rollt = -rollt;		//横方向符号調整
	pitcht=0.25*pitch;

	if(fwct>landF && fwct<=fwctEnd-landB ){

		// ************ UVC主計算 ************
		k	  = atan ((dyi-sw)/autoH );	//片脚の鉛直に対する開脚角
		kl	  = autoH/cos(k);			//前から見た脚投影長
		ks = k+rollt;					//開脚角に横傾き角を加算
		k  = kl*sin(ks);					//中点から横接地点までの左右距離
		dyi	  = k+sw;					//横方向UVC補正距離
		autoH = kl*cos(ks);				//K1までの高さ更新

		// **** UVC（前後） *****
		k 	  = atan( dxi/autoH );		//片脚のX駆動面鉛直から見た現時点の前後開脚角
		kl	  = autoH/cos(k);			//片脚のX駆動面鉛直から見た脚長
		ks	  = k+pitcht;				//振出角に前後傾き角を加算
		k	  = kl*sin(ks);				//前後方向UVC補正距離
		dxi	  = k;						//前後方向UVC補正距離
		autoH = kl*cos(ks);				//K1までの高さ更新

		// ************ UVC積分値リミット設定 ************
		if(dyi<  0)		dyi=   0;
		if(dyi> 45)		dyi=  45;
		if(dxi<-45)		dxi= -45;
		if(dxi> 45)		dxi=  45;

		// ************ 遊脚側を追従させる ************

		// **** 左右方向 *****
		dyis = dyi;					//遊脚Y目標値

		// **** 前後方向 *****
		dxis = -dxi;				//遊脚X目標値

		// ************ 両脚内側並行補正 ************
		if(jikuasi==0){		//両脚を並行以下にしない
			k = -sw+dyi;	//右足の外側開き具合
			ks=  sw+dyis;	//左足の外側開き具合
		}
		else{
			ks= -sw+dyi;	//左足の外側開き具合
			k =  sw+dyis;	//右足の外側開き具合
		}
		if(k+ks<0)dyis-=k+ks;	//遊脚を平衡に補正
	}
	roll =rb;
	pitch=pb;
}



////////////////////
//// 脚上げ操作 ////
////////////////////
void footUp(void){

	if( fwct>landF && fwct<=(fwctEnd-landB) )fh = fhMax * sin( M_PI*(fwct-landF)/(fwctEnd-(landF+landB)) );
	else									 fh = 0;
}



////////////////////
//// 横振り制御 ////
////////////////////
void swCont(void){
	float k,t;

	k=swMax*sinf(M_PI*fwct/fwctEnd);//sinカーブ
	t=atan( (fabs(dxi)+fabs(21.5*sin(wt)))/(dyi+21.5*cos(wt)-wt) );
	if(dxi>0)	swx =  k*sin(t);
	else		swx = -k*sin(t);
	swy=k*cos(t);
}



////////////////
//// 腕制御 ////
////////////////
void armCont(void){
	U1W[0]=510*dyis/70; //股幅に応じ腕を広げる
	if(U1W[0]<0)U1W[0]=0;
	U1W[1]=U1W[0];
}



////////////////////
//// 最終脚駆動 ////
////////////////////
void footCont(float x,float y,float h,int s){
//x:中点を0とする設置点前後方向距離（前+）
//y:中点を0とする設置点左右方向距離（右+）
//h:足首ロール軸を基準に股関節ロール軸までの地上高(Max194.5)
//s:軸足0/遊脚1、指定
//真正面からみたK1-A1間距離 	k = sqrt( x*x + h*h );
//K1-K0間40mm A0-A1間24.5mm 合計64.5mm

//骨格垂直真横からみたK0-A0間直角高さ k = k - 64.5
//K0-A0間直線距離 k = sqrt( x*x + k*k ); まとめると↓
//高さhの最大は40+65+65+24.5=194.5mm

	float k;

	k = sqrt(x*x+pow(sqrt(y*y+h*h)-64.5,2));	//K0-A0間直線距離
	if(k>129){
		autoH = sqrt(pow(sqrt(129*129-x*x)+64.5,2)-y*y);//高さ補正
		k=129;
	}

	x = CHG_SVA*asin(x/k);						//K0脚振り角度
	k = CHG_SVA*acos(k/130);					//膝曲げ角度
	if(k>1800)k=1800;							//60°Max

	if		(2*k-HW[s]> 100) k=(HW[s]+100)/2;	//回転速度最大 0.13s/60deg = 138count
	else if	(2*k-HW[s]<-100) k=(HW[s]-100)/2;
//	if(mode!=0 && jikuasi==s)HW[s]	= k*1.98;	//軸足のたわみを考慮
	if(mode!=0 && jikuasi==s)HW[s]	= k*2;
	else					 HW[s]	= k*2;
	K0W[s]	= k+x;
	A0W[s]	= k-x;

	k = CHG_SVA*atan(y/h);						//K1角度

	if		(k-K1W[s]> 100) k=K1W[s]+100;		//回転速度最大 0.13s/60deg = 138count
	else if	(k-K1W[s]<-100) k=K1W[s]-100;

	if(mode!=0 && jikuasi==s)K1W[s] = k;		//軸足のたわみを考慮
	else					 K1W[s] = k;
	A1W[s] = -k;
}



////////////////////
//// 統合脚駆動 ////
////////////////////
void feetCont1(float x0, float y0, float x1, float y1, int s){

	if(s==1){
		if(y0+21.5==0)		wt = 0;			//0除算回避
		else if(jikuasi==0){
			wt = 0.5*atan( x0/(y0+21.5) );	//腰回転角度
			wk=fabs(15.0*x0/45);			//45:最大歩幅 15:足内側最大移動量
		}
		else{
			wt = 0.5*atan( -x1/(y1+21.5) );	//腰回転角度
			wk=fabs(15.0*x1/45);
		}

		WESTW = wt*CHG_SVA;					//腰回転（+で上体右回転）
		K2W[0]= WESTW;
		K2W[1]=-WESTW;
	}

	if(jikuasi==0){
		footCont( x0,	y0-wk,	autoH   ,	0 );
		footCont( x1,	y1-wk,	autoH-fh,	1 );
	}
	else{
		footCont( x0,	y0-wk,	autoH-fh,	0 );
		footCont( x1,	y1-wk,	autoH   ,	1 );
	}
}



void feetCont2(int s){
	if(jikuasi==0)	feetCont1( dxi  -swx, dyi  -swy,	dxis -swx, dyis +swy ,s );
	else			feetCont1( dxis -swx, dyis +swy,	dxi  -swx, dyi  -swy ,s );
}



///////////////////////////
//// 周期カウンタの制御 ////
///////////////////////////
void counterCont(void){
	if(fwct>=fwctEnd){	//簡易仕様（固定周期）
		jikuasi^=1;
		fwct=0;

		fh=0;
		k=dyis;
		dyis=dyi;
		dyib=dyi;
		dyi=k;

		k=dxis;
		dxis=dxi;
		dxib=dxi;
		dxi=k;
	}
	else{
		fwct+=fwctUp;
		if(fwct>fwctEnd)fwct=fwctEnd;	//最大値補償
	}
}



//##################
//#### 歩行制御 ####
//##################
void walk(void){

	switch(mode){

//**** ① 開始、初期姿勢に以降 ****
case 710:
	movSv(&K0W[0],  661);
	movSv(&K1W[0],    0);
	movSv(&K2W[0],    0);
	movSv(& HW[0], 1322);
	movSv(&A0W[0],  661);
	movSv(&A1W[0],    0);//+30
	movSv(&U0W[0],-2700);
	movSv(&U1W[0],    0);
	movSv(&U2W[0],    0);
	movSv(& EW[0],    0);
	movSv(& WESTW,    0);

	movSv(&K0W[1],  661);
	movSv(&K1W[1],    0);
	movSv(&K2W[1],    0);
	movSv(& HW[1], 1322);
	movSv(&A0W[1],  661);
	movSv(&A1W[1],    0);//-30
	movSv(&U0W[1],-2760);
	movSv(&U1W[1],    0);
	movSv(&U2W[1],    0);
	movSv(& EW[1],    0);
	movSv(& HEADW,    0);

	if(motCt>0)--motCt;
	else{
		//// 角度補正用 ////
		p_ofs	=0;
		r_ofs	=0;
		ip		=0;
		ir		=0;
		ipb		=0;
		irb		=0;
		ipa		=0;
		ira		=0;

		//// UVC積分用 ////
		dxi		=0;
		dyi		=0;
		dxis	=0;
		dyis	=0;
		dxib	=0;
		dyib	=0;

		landF	=0;
		landB	=0;
		fwctEnd	=18;
		walkCt	=0;
		fwctUp	=1;
		fwct=1;
		autoH=HEIGHT;
		sw=0;
		swx=0;
		swy=0;
		jikuasi=1;

		//// 初期姿勢 ////
		footCont(0,0,HEIGHT,0);
		jikuasi=0;
		footCont(0,0,HEIGHT,1);
		mode=720;					//状態遷移
		sprintf( (char *)dsp,"mode=720\r\n" );
		printS((char *)dsp);
	}
	break;


//**** ② 開始時の傾斜角補正 ****
case 720:
	angAdj();
	if( ip==100 ){
		p_ofs=ipa/100;
		r_ofs=ira/100;
		mode=730;		//状態遷移
		sprintf( (char *)dsp,"\r\nmode=730 ** Ready ** Pa:%4d Po:%4d  Ra:%4d Ro:%4d\r\n",pitchs,p_ofs,rolls,r_ofs );
		printS((char *)dsp);
	}
	detAng();			//転倒検知
	break;


//**** ③ 初期姿勢における傾斜検知 ****
case 730:

	// **** offset値補正 ****
	if( ip>=50 ){
		ip=0;
		if(rolls >0) ++r_ofs;
		if(rolls <0) --r_ofs;
		if(pitchs>0) ++p_ofs;
		if(pitchs<0) --p_ofs;
	}
	else ++ip;

	feetCont2(1);

	if(	fabs(roll)>0.033 || fabs(pitch)>0.044 ){
		if(roll>0)	jikuasi=1;
		else		jikuasi=0;
		fwct=1;
		mode=740;					//状態遷移
		sprintf( (char *)dsp,"mode=740\r\n" );
		printS((char *)dsp);
		break;
	}
	sprintf( (char *)dsp,"P:%4d R:%4d C:%4d\r",pitchs,rolls,ip );
	printS((char *)dsp);
	break;


//**** ④ UVC動作開始 ****
case 740:
	uvc();					//UVCメイン制御
	uvcSub();				//UVCサブ制御
	footUp();				//脚上げによる股関節角算出
	feetCont2(1);			//脚駆動
	armCont();
	counterCont();			//周期カウンタの制御
	if(fwct==0){
		mode=750;			//状態遷移
		sprintf( (char *)dsp,"mode=750\r\n" );
		printS((char *)dsp);
	}
	detAng();			//転倒検知
	break;


//**** ⑤ UVC後、振動減衰待ち ****
case 750:
	feetCont2(1);
	if(	fwct>30 ){
		fwct=1;

		k=sqrt(0.5*dxis*dxis+dyis*dyis);	//移動量、前後方向は減少させる
		swMax=17+17*k/45;

		mode=760;		//状態遷移
		sprintf( (char *)dsp,"mode=760 %2d \r\n",(int)swMax );
		printS((char *)dsp);
		break;
	}
	else{
		sprintf( (char *)dsp,"C:%4d\r",(int)fwct );
		printS((char *)dsp);
		++fwct;
	}
	detAng();			//転倒検知
	break;


//**** ⑥ 回復動作 ****
case 760:
	landF=25;
	fwctEnd=landF+25;//15
	uvcSub2();			//UVCサブ制御
	footUp();			//脚上げによる股関節角算出
	swCont();			//横振り制御
	feetCont2(0);		//脚駆動
	armCont();
	counterCont();		//周期カウンタの制御5
	if(fwct==0){
		landF=0;
		dxi		=0;
		dyi		=0;
		dxis	=0;
		dyis	=0;
		dxib	=0;
		dyib	=0;
		landF	=0;
		landB	=0;
		fwctEnd	=18;
		walkCt	=0;
		fwctUp	=1;
		fwct=1;
		autoH=HEIGHT;
		sw=0;
		swx=0;
		swy=0;
		jikuasi=1;
		mode=770;			//状態遷移
		sprintf( (char *)dsp,"mode=770\r\n" );
		printS((char *)dsp);
	}
	detAng();			//転倒検知
	break;


//**** ⑦ 回復後、振動減衰待ち ****
case 770:
	feetCont2(0);
	if(	fwct>50 ){//50
		fwct=1;
		mode=730;		//状態遷移
		sprintf( (char *)dsp,"mode=730\r\n" );
		printS((char *)dsp);
		break;
	}
	else{
		sprintf( (char *)dsp,"C:%4d\r",(int)fwct );
		printS((char *)dsp);
		++fwct;
	}
	detAng();			//転倒検知
	break;


case 780:
	feetCont2(1);
	break;


case 790:
	break;

case 791:

	dxis=-dxi;
	dyis=dyi;

	feetCont2(1);
	break;


case 700:				//モニター
	sprintf( (char *)dsp,"R:%4d  %4d  RG:%4d  %4d\r\n", (int)(roll*1000),(int)(pitcht*1000), (int)roll_gyr,(int)pitch_gyr);
	printS((char *)dsp);
	break;
	}
}



//######################
//#### キー読込制御 ####
//######################
void keyCont(void){

	//////////////////////////
	////// TTYコマンド処理 //////
	//////////////////////////
	ff[0]=0;
	ff[1]=0;
	uart_rx (UART_COM, ff, 1,1);
	if(ff[0]!=0){
		sprintf( (char *)dsp,"%c \r\n",ff[0] );
		printS((char *)dsp);

		if(ff[0]==' ')keyMode=0;//キーモードリセット

		///////////////////////
		//// 基本入力モード ////
		///////////////////////
		if(keyMode==0){
			switch(ff[0]){
				case 'r':		//リセット
					motCt=100;
					mode=710;
					sprintf( (char *)dsp,"**** Reset ****\r\n" );
					printS((char *)dsp);
					break;

				case 'g':		//開始
					sprintf( (char *)dsp,"**** Go ****\r\n" );
					printS((char *)dsp);
					break;

				case 't':		//試験
					mode=790;
					sprintf( (char *)dsp,"**** angle Disp ****\r\n" );
					printS((char *)dsp);
					break;

				case 'y':		//試験
					mode=791;
					sprintf( (char *)dsp,"**** debug ****\r\n" );
					printS((char *)dsp);
					break;

//				case 'm':		//パラメタモード
//					keyMode=1;
//					break;
				case 'p':		//特定変数設定モード
					keyMode=5;
					break;
				case 'k':		//サーボ Kグループ
					keyMode=2;
					break;
				case 'u':		//サーボ Uグループ
					keyMode=3;
					break;
				case 'a':		//サーボ Aグループ
					keyMode=4;
					break;
				case 'h':		//サーボ Hグループ
					keyMode=50;
					break;
				case 'e':		//サーボ Eグループ
					keyMode=60;
					break;
				case 'z':		//サーボ HEAD
					keyMode=700;
					break;
				case 'w':		//サーボ WEST
					keyMode=800;
					break;
			}
		}


		///////////////////////////
		//// 特定変数設定モード ////
		///////////////////////////
		if(keyMode==5){
			switch(ff[0]){
				case '0':
					kn=0;
					goto dd2;
				case '1':
					kn=1;
					goto dd2;
				case '2':
					kn=2;
					goto dd2;
				case '3':
					kn=3;
					goto dd2;
				case '4':
					kn=4;
					goto dd2;
				case '5':
					kn=5;
					goto dd2;
				case '6':
					kn=6;
					goto dd2;
				case '7':
					kn=7;
					goto dd2;
				case '8':
					kn=8;
					goto dd2;
				case '9':
					kn=9;
					goto dd2;
				case '+':
				case '-':
					switch(kn){
					case 0:dxi			+= ff[0]=='+'?	1:	-1;		break;
					case 1:dyi			+= ff[0]=='+'?	1:	-1;		break;
					case 2:swMax		+= ff[0]=='+'?	1:	-1;		break;
					case 3:pitch_gyrg	+= ff[0]=='+'?	0.01:-0.01;	break;
					case 4:roll_gyrg	+= ff[0]=='+'?	0.01:-0.01;	break;
					case 5:fh			+= ff[0]=='+'?  1:  -1;		break;
					case 6:fhMax		+= ff[0]=='+'?	1:  -1;		break;
					case 7:walkCtLim	+= ff[0]=='+'?  1:  -1;		break;
					case 8:autoH		+= ff[0]=='+'?  1:  -1;		break;
					}
dd2:
					sprintf( (char *)dsp, "No:%d\r\n0 dx:%d\r\n1 dy:%d\r\n2 sw:%d\r\n3 pg:%d\r\n4 rg:%d\r\n5 fh:%d\r\n6 fh%d\r\n7  wc%d\r\n8 aH:%d\r\n"
						,kn 	,(int)dxi,(int)dyi,(int)swMax,(int)(pitch_gyrg*100),(int)(roll_gyrg*100),(int)fh,(int)fhMax,(int)walkCtLim,(int)autoH);
					printS((char *)dsp);
					break;
			}
		}


		/////////////////////////
		//// サーボ設定モード ////
		/////////////////////////
		if(keyMode==2){			//サーボ Kグループ
			switch(ff[0]){
				case '0':		//K0選択
					keyMode=20;
					break;
				case '1':		//K1選択
					keyMode=21;
					break;
				case '2':		//K2選択
					keyMode=22;
					break;
			}
		}
		if(keyMode==3){			//サーボ Uグループ
			switch(ff[0]){
				case '0':		//U0選択
					keyMode=30;
					break;
				case '1':		//U1選択
					keyMode=31;
					break;
				case '2':		//U2選択
					keyMode=32;
					break;
			}
		}
		if(keyMode==4){			//サーボ Aグループ
			switch(ff[0]){
				case '0':		//A0選択
					keyMode=40;
					break;
				case '1':		//A1選択
					keyMode=41;
					break;
			}
		}

		if(keyMode>=20&&keyMode<=60){		//サーボ K,U,A,N,Eグループ
			switch(ff[0]){
				case 'r':		//K0選択
					keyMode=keyMode*10;
					break;
				case 'l':		//K1選択
					keyMode=keyMode*10+1;
					break;
				case 'b':		//K2選択
					keyMode=keyMode*10+2;
					break;
			}
		}

		if(keyMode>=200&&keyMode<=800){		//サーボ K,U,A,N,Eグループ
			i=0;
			if(ff[0]=='+')i= 30;
			if(ff[0]=='-')i=-30;
			if(ff[0]=='+'||ff[0]=='-'){
				switch(keyMode){
				case 200:	K0W[0]+=i;	break;
				case 201:	K0W[1]+=i;	break;
				case 202:	K0W[0]+=i;K0W[1]+=i;break;
				case 210:	K1W[0]+=i;	break;
				case 211:	K1W[1]+=i;	break;
				case 212:	K1W[0]+=i;K1W[1]+=i;break;
				case 220:	K2W[0]+=i;	break;
				case 221:	K2W[1]+=i;	break;
				case 222:	K2W[0]+=i;K2W[1]+=i;break;

				case 300:	U0W[0]+=i;	break;
				case 301:	U0W[1]+=i;	break;
				case 302:	U0W[0]+=i;U0W[1]+=i;break;
				case 310:	U1W[0]+=i;	break;
				case 311:	U1W[1]+=i;	break;
				case 312:	U1W[0]+=i;U1W[1]+=i;break;
				case 320:	U2W[0]+=i;	break;
				case 321:	U2W[1]+=i;	break;
				case 322:	U2W[0]+=i;U2W[1]+=i;break;

				case 400:	A0W[0]+=i;	break;
				case 401:	A0W[1]+=i;	break;
				case 402:	A0W[0]+=i;A0W[1]+=i;break;
				case 410:	A1W[0]+=i;	break;
				case 411:	A1W[1]+=i;	break;
				case 412:	A1W[0]+=i;A1W[1]+=i;break;

				case 500:	HW[0]+=i;	break;
				case 501:	HW[1]+=i;	break;
				case 502:	HW[0]+=i;HW[1]+=i;break;

				case 600:	EW[0]+=i;	break;
				case 601:	EW[1]+=i;	break;
				case 602:	EW[0]+=i;EW[1]+=i;break;

				case 700:	HEADW+=i;	break;
				case 701:	HEADW+=i;	break;

				case 800:	WESTW+=i;	break;
				case 801:	WESTW+=i;	break;
				}
				sprintf( (char *)dsp,    "Mode=%d\r\n",modeNxt );
				printS((char *)dsp);
				sprintf( (char *)dsp,    "K0:%7d %7d K1:%7d %7d K2:%7d %7d \r\n",K0W[0],K0W[1],K1W[0],K1W[1],K2W[0],K2W[1] );
				printS((char *)dsp);
				sprintf( (char *)dsp,    "H:%7d %7d \r\n",HW[0] ,HW[1]  );
				printS((char *)dsp);
				sprintf( (char *)dsp,    "A0:%7d %7d A1:%7d %7d \r\n\r\n",A0W[0],A0W[1],A1W[0],A1W[1] );
				printS((char *)dsp);
				sprintf( (char *)dsp,    "U0:%7d %7d U1:%7d %7d U2:%7d %7d \r\n",U0W[0],U0W[1],U1W[0],U1W[1],U2W[0],U2W[1] );
				printS((char *)dsp);
				sprintf( (char *)dsp,    "E:%7d %7d \r\n\r\n",EW[0] ,EW[1]  );
				printS((char *)dsp);
				sprintf( (char *)dsp,    "HD:%7d WT:%7d \r\n",HEADW ,WESTW  );
				printS((char *)dsp);
			}
		}
	}
}




// **************************************************************************
// *************** Main routine *********************************************
// **************************************************************************
int main(void){

	K0W[0]=0;			//股関節前後方向右書込用
	K1W[0]=0;			//股関節横方向右書込用
	K2W[0]=0;			//股関節横方向右書込用
	HW [0]=0;			//膝関節右書込用
	A0W[0]=0;			//足首上下方向右書込用
	A1W[0]=0;			//足首横方向右書込用
	U0W[0]=-5400;			//肩前後方向右書込用
	U1W[0]=0;			//肩横後方向右書込用
	U2W[0]=0;			//肩ヨー向右書込用
	 EW[0]=0;			//肘右書込用
	 WESTW=0;			//腰回転書込用

	K0W[1]=0;			//股関節前後方向左書込用
	K1W[1]=0;			//股関節横方向左書込用
	K2W[1]=0;			//股関節横方向左書込用
	HW [1]=0;			//膝関節左書込用
	A0W[1]=0;			//足首上下方向左書込用
	A1W[1]=0;			//足首横方向左書込用
	U0W[1]=-5400;			//肩前後方向左書込用
	U1W[1]=0;			//肩横後方向左書込用
	U2W[1]=0;			//肩ヨー向左書込用
	 EW[1]=0;			//肘左書込用
	 HEADW=0;			//頭回転書込用


	///////////////////////
	//// タイマの初期化 ////
	///////////////////////
	timer_init(TIMER, TIMER_MODE_TIMER32, 1000000);//取り敢えず周期１秒に設定
	delay(500);//wait 500ms この時間待たないとダメ


	///////////////////
	//// PIO初期化 ////
	///////////////////
	pio_init(PIO_LED1, PIO_SET_OUT);// PIO(LED1)を出力に設定
	pio_init(PIO_LED2, PIO_SET_OUT);// PIO(LED2)を出力に設定
	pio_init(PIO_T1, PIO_SET_IN);	// PIO(T1)を入力に設定
	pio_init(PIO_T2, PIO_SET_OUT);	// PIO(T2)を出力に設定
	pio_init(PIO_T3, PIO_SET_IN);	// PIO(T3)を入力に設定
	pio_init(PIO_T4, PIO_SET_IN);	// PIO(T4)を入力に設定
	pio_init(PIO_T5, PIO_SET_OUT);	// PIO(T5)を出力に設定
	pio_init(PIO_T6, PIO_SET_OUT);	// PIO(T6)を出力に設定
	pio_init(PIO_SW1, PIO_SET_IN);	// PIO(SW1)を入力に設定
	pio_init(PIO_SW2, PIO_SET_IN);	// PIO(SW2)を入力に設定
	pio_write (PIO_LED2, HIGH);		// 緑 OFF


	/////////////////////////////
	//// シリアルポート初期化 ////
	/////////////////////////////
	uart_init(UART_COM, UART, BR115200, 8, PARITY_NONE);
	i2c_init ( 400000, I2C_MASTER );

	sio_init (UART_SIO1, BR1250000);	// SIOの初期化
	sio_init (UART_SIO2, BR1250000);	// SIOの初期化
	sio_init (UART_SIO3, BR1250000);	// SIOの初期化
	sio_init (UART_SIO4, BR1250000);	// SIOの初期化


	/////////////////////////////
	//// アナログポート初期化 ////
	/////////////////////////////
	ad_init(PIO_AD1, SWEEP);		//アナログポート１設定
	ad_init(PIO_AD2, SWEEP);		//アナログポート２設定
	dac_init();						//アナログ出力設定


	////////////////////////
	//// 更新日時を表示 ////
	////////////////////////
	sprintf((char *)dsp,"Version: %s %s\r\n", __DATE__, __TIME__);
	printS((char *)dsp);


	/////////////////////////
	//// BMO055 開始処理 ////
	/////////////////////////
	//// Make sure we have the right device ////
	if(read8(0) !=0xA0 ){		// BNO055_ID
		delay(1000);			// hold on for boot
		if(read8(0) !=0xA0 ){	// BNO055_ID
			printS("*** NG1 ***\r\n");
			return false;		// still not? ok bail
		}
	}
	//// Switch to config mode (just in case since this is the default) ////
	write8(0X3D, 0);			//BNO055_OPR_MODE_ADDR  //OPERATION_MODE_CONFIG
	delay(30);
	//// Reset ////
	write8(0X3F, 0x20);			//BNO055_SYS_TRIGGER_ADDR
	delay(500);
	while (read8(0) != 0xA0){	//BNO055_CHIP_ID_ADDR  //BNO055_ID
		delay(1000);			// hold on for boot
		if(read8(0) != 0xA0 ){			// BNO055_ID
			printS("*** NG2 ***\r\n");
			return false;		// still not? ok bail
		}
	}
	delay(50);
	//// Set to normal power mode ////
	write8(0X3E, 0X00);			//BNO055_PWR_MODE_ADDR  //POWER_MODE_NORMAL
	delay(10);
	write8(0X07, 0);			//BNO055_PAGE_ID_ADDR
	write8(0X3F, 0);			//BNO055_SYS_TRIGGER_ADDR
	delay(10);
	//// Set the requested operating mode (see section 3.3) ////
	//  _mode = 0X0C;			//OPERATION_MODE_NDOF
	write8(0X3D, 0X0C);			//BNO055_OPR_MODE_ADDR  //mode
	delay(1000);
	//// Use external crystal for better accuracy ////
	write8(0X3D, 0);			//BNO055_OPR_MODE_ADDR  //OPERATION_MODE_CONFIG
	delay(50);
	write8(0X07, 0);			//BNO055_PAGE_ID_ADDR
	write8(0x0, 0x80);			//BNO055_SYS_TRIGGER_ADDR
	delay(10);
	//// Set the requested operating mode (see section 3.3) ////
	write8(0X3D, 0X0C);			//BNO055_OPR_MODE_ADDR  //modeback
	delay(50);
	printS("*** BNO055 INIT OK ***\r\n");


	/////////////////////////////
	//// サーボ現在角度 ////
	/////////////////////////////
	i=ics_set_pos ( UART_SIO2, 1, 0 );	//U0Rバンザイ位置
	U0W[0]=-i+4735;
	i=ics_set_pos ( UART_SIO2, 2, 0 );	//U1R +2700
	U1W[0]=-i+10110;
	i=ics_set_pos ( UART_SIO2, 3, 0 );	//U2R
	U2W[0]=-i+7500;
	i=ics_set_pos ( UART_SIO2, 4, 0 );	//ER
	EW [0]=i-4800;
	i=ics_set_pos ( UART_SIO4, 1, 0 );	//U0Lバンザイ位置
	U0W[1]=i-9320;
	i=ics_set_pos ( UART_SIO4, 2, 0 );	//U1L -2700
	U1W[1]=i-4850;
	i=ics_set_pos ( UART_SIO4, 3, 0 );	//U2L
	U2W[1]=i-7500;
	i=ics_set_pos ( UART_SIO4, 4, 0 );	//EL
	EW [1]=-i+10150;
	i=ics_set_pos ( UART_SIO1, 5, 0 );	//K2R
	K2W[0]=-i+7500;
	i=ics_set_pos ( UART_SIO1, 6, 0 );	//K1R
	K1W[0]=-i+7470;
	i=ics_set_pos ( UART_SIO1, 7, 0 );	//K0R
	K0W[0]=-i+7500;
	i=ics_set_pos ( UART_SIO1, 8, 0 );	//HR +1760
	HW [0]=-i+9260;
	i=ics_set_pos ( UART_SIO1, 9, 0 );	//A0R +350
	A0W[0]=i-7910;
	i=ics_set_pos ( UART_SIO1,10, 0 );	//A1R
	A1W[0]=i-7585;
	i=ics_set_pos ( UART_SIO3, 5, 0 );	//K2L
	K2W[1]=i-7500;
	i=ics_set_pos ( UART_SIO3, 6, 0 );	//K1L
	K1W[1]=i-7500;
	i=ics_set_pos ( UART_SIO3, 7, 0 );	//K0L
	K0W[1]=i-7500;
	i=ics_set_pos ( UART_SIO3, 8, 0 );	//HL -1760
	HW [1]=i-5740;
	i=ics_set_pos ( UART_SIO3, 9, 0 );	//A0L -350
	A0W[1]=-i+7100;
	i=ics_set_pos ( UART_SIO3,10, 0 );	//A1L
	A1W[1]=-i+7530;
	i=ics_set_pos ( UART_SIO4, 0, 0 );	//HEADL
	HEADW=i-7500;
	i=ics_set_pos ( UART_SIO2, 0, 0 );	//WESTR
	WESTW=i-7500;


	/////////////////////////////
	//// サーボストレッチ設定 ////
	/////////////////////////////
//	ics_set_param ( UART_SIO1, 7,ICS_STRC_SC,250);	//K0R
//	ics_set_param ( UART_SIO3, 7,ICS_STRC_SC,250);	//K0L

	ics_set_param ( UART_SIO2, 1,ICS_STRC_SC,20);	//U0R
	ics_set_param ( UART_SIO4, 1,ICS_STRC_SC,20);	//U0L

	ics_set_param ( UART_SIO2, 2,ICS_STRC_SC,20);	//U1R
	ics_set_param ( UART_SIO4, 2,ICS_STRC_SC,20);	//U1L

	ics_set_param ( UART_SIO2, 4,ICS_STRC_SC,20);	//ER
	ics_set_param ( UART_SIO4, 4,ICS_STRC_SC,20);	//EL


	////////////////////////
	//// 10msタイマ開始 ////
	////////////////////////
	timer_write(TIMER,1000000);
	timer_start(TIMER);


	////////////////////
	//// 変数初期化 ////
	////////////////////

	LEDct=0;	//LED点灯カウンタ

	tBak=0;
	pitchi=0;
	tNow=0;

	p_ofs=0;
	r_ofs=0;
	ir=0;
	ip=0;
	irb=0;
	ipb=0;

	kn=0;

	motCt=100;
	keyMode=0;
	cycle=10000;
	mode=710;
	pitch_gyrg=0.08;
	roll_gyrg=0.1;

	swMax=25;//22
	fhMax=35;
	walkCtLim=3;



//----------------------------------------------------------------------------------
	////////////////////////////////////////////////
	//////////////////  MAIN LOOP  /////////////////
	////////////////////////////////////////////////
top:
	//////////////////////
	//// 10ms待ち処理 ////
	//////////////////////
pio_write (PIO_T2, HIGH);	//OFF(wait時間確認)

	do{
		tNow=timer_read(TIMER);
	}while(tNow<cycle);

	if(tNow>cycle+10){
		sprintf( (char *)dsp,"************** %d \r\n",(int)tNow);
		printS ( (char *)dsp );
	}
	timer_start(TIMER);


	////////////////////
	//// サーボ設定 ////
	////////////////////

	//// 関節リミット ////
	if(K1W[0]> 800)K1W[0]	= 800;
	if(K1W[0]<-450)K1W[0]	=-450;
	if(K1W[1]> 800)K1W[1]	= 800;
	if(K1W[1]<-450)K1W[1]	=-450;
	if(A0W[0]> 3500)A0W[0]	= 3500;
	if(A0W[0]<-3500)A0W[0]	=-3500;
	if(A0W[1]> 3500)A0W[1]	= 3500;
	if(A0W[1]<-3500)A0W[1]	=-3500;
	if(A1W[0]> 420)A1W[0]	= 420;	//添付品曲加工+アルミソールでの実験結果
	if(A1W[0]<-900)A1W[0]	=-900;	//添付品曲加工+アルミソールでの実験結果
	if(A1W[1]> 420)A1W[1]	= 420;	//添付品曲加工+アルミソールでの実験結果
	if(A1W[1]<-900)A1W[1]	=-900;	//添付品曲加工+アルミソールでの実験結果

	//// ICSデータ送受信 ////
	//// 注意ポジションデータが10500を超えるとサーボが応答しない（間欠反応）
	//// 注意ポジションデータが 3600以下では反応はするが動かない

	pio_write (PIO_T2, LOW );	//ON
	ics_set_pos    ( UART_SIO1, 5, 7560 -(K2W[0])-60 );	//K2R
	tNow=timer_read(TIMER);if(tNow>10010){sprintf( (char *)dsp,"K2R**** %d \r\n",(int)tNow);printS ( (char *)dsp );}

	pio_write (PIO_T2, HIGH);	//OFF
	ii=ics_set_pos ( UART_SIO2, 1, 4735 -(U0W[0]) );	//U0Rバンザイ位置
	tNow=timer_read(TIMER);if(tNow>10010){sprintf( (char *)dsp,"U0R**** %d \r\n",(int)tNow);printS ( (char *)dsp );}
	U0R=-(ii-4735);

	pio_write (PIO_T2, LOW );	//ON
	ics_set_pos    ( UART_SIO3, 5, 7500 +(K2W[1])-90 );	//K2L
	tNow=timer_read(TIMER);if(tNow>10010){sprintf( (char *)dsp,"K2L**** %d \r\n",(int)tNow);printS ( (char *)dsp );}

	pio_write (PIO_T2, HIGH);	//OFF
	ii=ics_set_pos ( UART_SIO4, 1, 9320 +(U0W[1])-60 );	//U0Lバンザイ位置
	tNow=timer_read(TIMER);if(tNow>10010){sprintf( (char *)dsp,"U0L**** %d \r\n",(int)tNow);printS ( (char *)dsp );}
	U0L=(ii-9230);

	pio_write (PIO_T2, LOW );	//ON
	ics_set_pos    ( UART_SIO1, 6, 7470 -(K1W[0])-30 );	//K1R
	tNow=timer_read(TIMER);if(tNow>10010){sprintf( (char *)dsp,"K1R**** %d \r\n",(int)tNow);printS ( (char *)dsp );}

	pio_write (PIO_T2, HIGH);	//OFF
	ii=ics_set_pos ( UART_SIO2, 2,10110 -(U1W[0]) );	//U1R +2700
	tNow=timer_read(TIMER);if(tNow>10010){sprintf( (char *)dsp,"U1R**** %d \r\n",(int)tNow);printS ( (char *)dsp );}
	U1R=-(ii-10110);

	pio_write (PIO_T2, LOW );	//ON
	ics_set_pos    ( UART_SIO3, 6, 7650 +(K1W[1])-90 );//K1L
	tNow=timer_read(TIMER);if(tNow>10010){sprintf( (char *)dsp,"K1L**** %d \r\n",(int)tNow);printS ( (char *)dsp );}

	pio_write (PIO_T2, HIGH);	//OFF
	ii=ics_set_pos ( UART_SIO4, 2, 4850 +(U1W[1]) );	//U1L -2700
	tNow=timer_read(TIMER);if(tNow>10010){sprintf( (char *)dsp,"U1L**** %d \r\n",(int)tNow);printS ( (char *)dsp );}
	U1L=ii-4850;


	pio_write (PIO_T2, LOW );	//ON
	ii=ics_set_pos ( UART_SIO1, 7, 7480 -(K0W[0])-30 );	//K0R
	tNow=timer_read(TIMER);if(tNow>10010){sprintf( (char *)dsp,"K0R**** %d \r\n",(int)tNow);printS ( (char *)dsp );}
	K0R=7510-ii;

	pio_write (PIO_T2, HIGH);	//OFF
	ics_set_pos    ( UART_SIO2, 3, 7500 -(U2W[0]) );	//U2R
	tNow=timer_read(TIMER);if(tNow>10010){sprintf( (char *)dsp,"U2R**** %d \r\n",(int)tNow);printS ( (char *)dsp );}

	pio_write (PIO_T2, LOW );	//ON
	ii=ics_set_pos ( UART_SIO3, 7, 7500 +(K0W[1]) );	//K0L
	tNow=timer_read(TIMER);if(tNow>10010){sprintf( (char *)dsp,"K0L**** %d \r\n",(int)tNow);printS ( (char *)dsp );}
	K0L=ii-7500;

	pio_write (PIO_T2, HIGH);	//OFF
	ics_set_pos    ( UART_SIO4, 3, 7500 +(U2W[1]) );	//U2L
	tNow=timer_read(TIMER);if(tNow>10010){sprintf( (char *)dsp,"U2L**** %d \r\n",(int)tNow);printS ( (char *)dsp );}

	pio_write (PIO_T2, LOW );	//ON
	ics_set_pos    ( UART_SIO1, 8, 9320 -(HW [0]) );	//HR +1760
	tNow=timer_read(TIMER);if(tNow>10010){sprintf( (char *)dsp,"HR**** %d \r\n",(int)tNow);printS ( (char *)dsp );}

	pio_write (PIO_T2, HIGH);	//OFF
	ics_set_pos    ( UART_SIO2, 4, 4800 +(EW [0]+i) );	//ER
	tNow=timer_read(TIMER);if(tNow>10010){sprintf( (char *)dsp,"ER**** %d \r\n",(int)tNow);printS ( (char *)dsp );}

	pio_write (PIO_T2, LOW );	//ON
	ics_set_pos    ( UART_SIO3, 8, 5770 +(HW [1])-120 );	//HL -1760
	tNow=timer_read(TIMER);if(tNow>10010){sprintf( (char *)dsp,"HL**** %d \r\n",(int)tNow);printS ( (char *)dsp );}

	pio_write (PIO_T2, HIGH);	//OFF
	ics_set_pos    ( UART_SIO4, 4,10150 -(EW [1]+i) );	//EL
	tNow=timer_read(TIMER);if(tNow>10010){sprintf( (char *)dsp,"EL**** %d \r\n",(int)tNow);printS ( (char *)dsp );}


	i=pitch_gyrg*pitch_gyr;
	pio_write (PIO_T2, LOW );	//ON
	ics_set_pos    ( UART_SIO1, 9, 7870-10 +(A0W[0]) + i+60 );	//A0R

	pio_write (PIO_T2, HIGH);	//OFF
	ics_set_pos    ( UART_SIO2, 0, 7500 +(WESTW ) );			//WESTR

	pio_write (PIO_T2, LOW );	//ON
	ics_set_pos    ( UART_SIO3, 9, 7100    -(A0W[1]) - i );	//A0L

	pio_write (PIO_T2, HIGH);	//OFF
	ics_set_pos    ( UART_SIO4, 0, 7500 +(HEADW ) );			//HEADL

	i=roll_gyrg*roll_gyr;
	pio_write (PIO_T2, LOW );	//ON
	ics_set_pos    ( UART_SIO1,10, 7470 +(A1W[0]) - i-30 );		//A1R

	pio_write (PIO_T2, HIGH);	//OFF
	ics_set_pos    ( UART_SIO3,10, 7560 -(A1W[1]) - i-60 );		//A1L


	/////////////////////
	//// IMU読取処理 ////
	/////////////////////
	//      +----------+
	//      |         *| RST   PITCH  ROLL  HEADING
	//  ADR |*        *| SCL
	//  INT |*        *| SDA     ^            /->
	//  PS1 |*        *| GND     |            |
	//  PS0 |*        *| 3VO     Y    Z-->    \-X
	//      |         *| VIN
	//      +----------+

	//現状　2020 2/6    roll : 右側傾き（2879） 左側傾き（-2879）閾値2879
	//                  pitch: 前側傾き（+）    後側傾き（-）    補正値20

	readLen(0X1A, 6);	//絶対角度読込(degの16倍表示)
	yaw	   = ((int16_t)ff[0]) | (((int16_t)ff[1]) << 8); //直立右回転で +
	pitchs = ((int16_t)ff[2]) | (((int16_t)ff[3]) << 8); //直立前傾で   -
	rolls  = ((int16_t)ff[4]) | (((int16_t)ff[5]) << 8); //直立右傾斜で +
	if(rolls>0)	rolls= 2879-rolls;
	else		rolls=-2879-rolls;

	pitchs -= p_ofs;	//補正
	rolls  -= r_ofs;	//補正

	pitch = (float)pitchs*(M_PI/(180.0*16.0));	//radに変換
	roll  = (float)rolls *(M_PI/(180.0*16.0));	//radに変換


	readLen(0X14, 6);	//角速度読込 ※rollとyawが取説と実際が逆
	roll_gyr 	= ((int16_t)ff[0]) | (((int16_t)ff[1]) << 8);	//直立右傾斜で ＋
	pitch_gyr	= ((int16_t)ff[2]) | (((int16_t)ff[3]) << 8);	//直立前傾で   －
	yaw_gyr		= ((int16_t)ff[4]) | (((int16_t)ff[5]) << 8);	//直立右回転で ＋


	pio_write (PIO_T2, LOW );	//ON


	keyCont();


	walk();


	///////////////
	//// 頭LED ////
	///////////////
	++LEDct;
	if( LEDct > 100 )LEDct = -100;				//IMU Ready

	if( mode<=720 && LEDct > 10 )LEDct = -10;	//IMU not Ready

	if( LEDct > 0   ){
		dac_write (0xffff);
		pio_write (PIO_LED1, LOW );	//ON
	}
	else{
		dac_write (0);
		pio_write (PIO_LED1, HIGH);	//OFF
	}

	goto top;
}
