
s1 = id
s2 = "dataset/11/test5.jpg"

filename = 'report.html'
f = open(filename,'w')
message = """
<html>
<head></head>
<body>
<div class="main">
<div class="head">
<p class="head_content">Prediction Reports</p>
</div>
<div >
<p >"""+s1+"""</p>
</div>
<div class="label">
<div class="labelbox">
<p>
INPUT_resize(1536*1536)
</p>
</div>
<div class="labelbox">
<p>
PATCH_3*3(512*512)
</p>
</div>
</div>
<div class="imgcontainer">
<div class="input">
<img src=\""""+s2+"""\" width="300px" height="300px" margin-top="40px"/>
</div>
<div class="crop">
<div class="block9">
<div class="patchs">
<div class="patch">
<img src="dataset/sp9/0_0.jpg" width="100px" height="100px"/>
<span>Block 0_0</span>
</div>
<div class="patch">
<img src="dataset/sp9/0_1.jpg" width="100px" height="100px"/>
<span>Block 0_1</span>
</div>
<div class="patch">
<img src="dataset/sp9/0_2.jpg" width="100px" height="100px"/>
<span>Block 0_2</span>
</div>
</div>
<div class="patchs">
<div class="patch">
<img src="dataset/sp9/1_0.jpg" width="100px" height="100px"/>
<span>Block 1_0</span>
</div>
<div class="patch">
<img src="dataset/sp9/1_1.jpg" width="100px" height="100px"/>
<span>Block 1_1</span>
</div>
<div class="patch">
<img src="dataset/sp9/1_2.jpg" width="100px" height="100px"/>
<span>Block 1_2</span>
</div>
</div>
<div class="patchs">
<div class="patch">
<img src="dataset/sp9/2_0.jpg" width="100px" height="100px"/>
<span>Block 2_0</span>
</div>
<div class="patch">
<img src="dataset/sp9/2_1.jpg" width="100px" height="100px"/>
<span>Block 2_1</span>
</div>
<div class="patch">
<img src="dataset/sp9/2_2.jpg" width="100px" height="100px"/>
<span>Block 2_2</span>
</div>
</div>
</div>
</div>
</div>
<div class="title">
<span>Global Confidence Probability</span>
</div>
<div class="text">
<span class="discribe">
<br/>
During the image preprocessing process, a global <br/>
confidence probability will be assigned to each patch<br/>
according to the percentage of the cavity area<br/><br/>
</span>
</div>
<div class="title">
<span>Model Prediction - Swin Transformer (RBG PATTERN)</span>
</div>
<div class="text">
<span class="discribe">
<br/>
Possibility - means the chance of this block to be Metastatic-Tumor<br/>
Possibility = Confidence * Model_acc * Prediction_Possibility<br/>
<br/>
</span>
<span>Possibility of Block 0_0 = """+s3+"""</span>
<span>Possibility of Block 0_1 = """+s4+"""</span>
<span>Possibility of Block 0_2 = """+s5+"""</span>
<span>Possibility of Block 1_0 = """+s6+"""</span>
<span>Possibility of Block 1_1 = """+s7+"""</span>
<span>Possibility of Block 1_2 = """+s8+"""</span>
<span>Possibility of Block 2_0 = """+s9+"""</span>
<span>Possibility of Block 2_1 = """+s10+"""</span>
<span>Possibility of Block 2_2 = """+s11+"""</span>
<br/>
</div>
<div class="title">
<span>Model Prediction - Swin Transformer (BINARY PATTERN)</span>
</div>
<div class="text">
<span class="discribe">
<br/>
Possibility - means the chance of this block to be Metastatic-Tumor<br/>
Possibility = Confidence * Model_acc * Prediction_Possibility<br/>
<br/>
</span>
<span>Possibility of Block 0_0 = """+s12+"""</span>
<span>Possibility of Block 0_1 = """+s13+"""</span>
<span>Possibility of Block 0_2 = """+s14+"""</span>
<span>Possibility of Block 1_0 = """+s15+"""</span>
<span>Possibility of Block 1_1 = """+s16+"""</span>
<span>Possibility of Block 1_2 = """+s17+"""</span>
<span>Possibility of Block 2_0 = """+s18+"""</span>
<span>Possibility of Block 2_1 = """+s19+"""</span>
<span>Possibility of Block 2_2 = """+s20+"""</span>
<br/>
</div>
<div class="title">
<span>Heat Map - Attention of Model (RBG PATTERN)</span>
</div>
<div class="text">
<div class="graph">
<br/>
<img src="dataset/sp9/heat_map1.jpg" width="800px" height="200px" />
<br/>
</div>
</div>
<div class="title">
<span>Heat Map - Attention of Model (BINARY PATTERN)</span>
</div>
<div class="text">
<div class="graph">
<br/>
<img src="dataset/sp9/heat_map2.jpg" width="800px" height="200px" />
<br/>
</div>
</div>
<div class="title">
<span>Probability Sharing - Consider the neighbor node information</span>
</div>
<div class="text">
<span class="discribe">
<br/>
Consider the whole image as a graph just as we do in GNN<br/>
After a few iteration, we achieve local convergence <br/>
<br/>
</span>
<div class="graph">
<img src="dataset/graph.png" width="300px" height="300px" border="2 solid red"/>
<div class="prob">
<span>Average Possibility Of RBG PATTERN ~ """+s21+"""</span>
<br/>
<br/>
<br/>
<span>Average Possibility Of BINARY PATTERN ~ """+s22+"""</span>
</div>
</div>
</div>
<div class="title">
<span>Node Voting - Combine different pattern</span>
</div>
<div class="text">
<span class="discribe">
<br/>
The ultimate outcome is the combination of different prediction models and a trained threshold.<br/>
<br/>
</span>
<div class="graph">
<div class="prob">
<span>Prediction Result: """+s23+"""</span>
<span class="note">"""+s24+"""</span>
<br/>
<br/>
</div>
</div>
</div>
</div>
</body>
<style>
.main {
max-width: 1000px;
display: flex;
flex-direction: column;
align-items: center;
}
.head {
width: 100%;
height: 50px;
display: flex;
align-content: center;
align-items: center;
justify-content: center;
color: black;
}
.head_content {
font-weight: bold;
font-size: 28px;
}
.label{
display: flex;
width:100%;
background-color: #f9f9f9;
}
.labelbox{
display:flex ;
width:50%;
justify-content: center;
font-weight: bold;
}
.imgcontainer {
width: 100%;
display: flex;
background-color: #f9f9f9;
}
.input {
width: 50%;
display: flex;
justify-content: center;
align-items: center;
flex-direction: column;
}
.crop {
width: 50%;
display: flex;
justify-content: center;
align-items: center;
flex-direction: column;
}
.block9{
display: flex;
height:400px;
width:400px;
flex-direction: row;
align-items: center;
}
.patchs{
width:100%;
height:100%;
display: flex;
flex-direction: column;
justify-content: space-around;
}
.patch{
display: flex;
flex-direction: column;
align-content: center;
justify-content: center;
align-items: center;
}
.title{
width:100%;
height:50px;
font-weight: bold;
font-size:18px;
display: flex;
justify-content: center;
align-items: center;
background-color: #ececec;
}
.text{
display: flex;
width:100%;
/*height: 350px;*/
background-color: #f9f9f9;
text-align:center;
justify-content: center;
line-height: 1.5;
align-items: center;
flex-direction: column;
}
.discribe{
color: #9a9a9a;
}
.graph{
width:100%;
display: flex;
flex-direction: row;
justify-content: space-evenly;
align-items: center;
}
.prob{
display: flex;
height: 100%;
flex-direction: column;
}
.note{
font-size: 5px;
}
</style>
</html>
"""
f.write(message)
f.close()